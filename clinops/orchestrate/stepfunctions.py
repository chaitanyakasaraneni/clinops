"""
AWS Step Functions pipeline builder for clinops workflows.

Step Functions is a natural fit for clinical data pipelines: each stage
(ingest → preprocess → window → split → train) is an independent Lambda or
ECS task, failures can be retried automatically, and the execution history
provides a full audit trail — often required for clinical research data
governance.

This module provides a lightweight builder that constructs a sequential
Step Functions state machine definition from a list of :class:`PipelineStep`
objects. The state machine can be inspected as a dict (Amazon States Language),
deployed to AWS, or executed with a JSON payload.

Requires ``pip install clinops[aws]``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PipelineStep:
    """
    A single task in a Step Functions state machine.

    Parameters
    ----------
    name:
        Human-readable step name. Used as the state name in the ASL
        definition — must be unique within a pipeline.
    resource:
        ARN of the Lambda function, ECS task, or Step Functions activity
        that executes this step. See AWS docs for supported resource ARN
        formats.
    parameters:
        Static input parameters merged into the state's ``Parameters``
        block. Dynamic input from the previous step's output is also
        available via ``$$.Execution.Input`` or ``States.JsonMerge``.
    retry_attempts:
        Number of times to retry on ``Lambda.ServiceException``,
        ``Lambda.TooManyRequestsException``, or ``States.TaskFailed``.
        Default 3.
    timeout_seconds:
        Maximum execution time for this step in seconds. Default 3600 (1h).
    comment:
        Optional human-readable description embedded in the ASL state.

    Examples
    --------
    >>> step = PipelineStep(
    ...     name="Preprocess",
    ...     resource="arn:aws:lambda:us-east-1:123456789:function:clinops-preprocess",
    ...     parameters={"max_null_rate": 0.5},
    ...     retry_attempts=2,
    ... )
    """

    name: str
    resource: str
    parameters: dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3
    timeout_seconds: int = 3600
    comment: str = ""

    def to_asl_state(self, next_state: str | None) -> dict[str, Any]:
        """
        Render this step as an Amazon States Language Task state dict.

        Parameters
        ----------
        next_state:
            Name of the following state, or None if this is the terminal state.

        Returns
        -------
        dict[str, Any]
            ASL Task state definition.
        """
        state: dict[str, Any] = {
            "Type": "Task",
            "Resource": self.resource,
            "TimeoutSeconds": self.timeout_seconds,
            "Retry": [
                {
                    "ErrorEquals": [
                        "Lambda.ServiceException",
                        "Lambda.TooManyRequestsException",
                        "States.TaskFailed",
                    ],
                    "IntervalSeconds": 2,
                    "MaxAttempts": self.retry_attempts,
                    "BackoffRate": 2.0,
                }
            ],
        }
        if self.comment:
            state["Comment"] = self.comment
        if self.parameters:
            state["Parameters"] = self.parameters
        if next_state is not None:
            state["Next"] = next_state
        else:
            state["End"] = True
        return state


class StepFunctionsPipeline:
    """
    Build, deploy, and execute an AWS Step Functions state machine.

    Constructs a sequential pipeline from :class:`PipelineStep` objects.
    The generated state machine runs steps in the order they were added,
    passing each step's output as the input to the next.

    Requires ``pip install clinops[aws]``.

    Parameters
    ----------
    name:
        State machine name. Used as the AWS resource name.
    role_arn:
        ARN of the IAM role that Step Functions assumes to invoke each
        step's resource (Lambda, ECS, etc.).
    region:
        AWS region. Default ``"us-east-1"``.
    comment:
        Human-readable description embedded in the state machine definition.
    profile:
        AWS credentials profile name. If None, uses the default credential
        chain.

    Examples
    --------
    >>> pipeline = StepFunctionsPipeline(
    ...     name="clinops-daily-ingest",
    ...     role_arn="arn:aws:iam::123456789:role/StepFunctionsRole",
    ... )
    >>> pipeline.add_step(PipelineStep(
    ...     name="Ingest",
    ...     resource="arn:aws:lambda:us-east-1:123456789:function:clinops-ingest",
    ... ))
    >>> pipeline.add_step(PipelineStep(
    ...     name="Preprocess",
    ...     resource="arn:aws:lambda:us-east-1:123456789:function:clinops-preprocess",
    ... ))
    >>> print(pipeline.definition_json())   # inspect before deploying
    >>> arn = pipeline.deploy()
    >>> execution_arn = pipeline.execute({"date": "2024-01-01"})
    """

    def __init__(
        self,
        name: str,
        role_arn: str,
        region: str = "us-east-1",
        comment: str = "",
        profile: str | None = None,
    ) -> None:
        self.name = name
        self.role_arn = role_arn
        self.region = region
        self.comment = comment
        self.profile = profile
        self._steps: list[PipelineStep] = []
        self._client: Any = None

    def add_step(self, step: PipelineStep) -> StepFunctionsPipeline:
        """
        Append a step to the pipeline.

        Parameters
        ----------
        step:
            Step to add. Step names must be unique within the pipeline.

        Returns
        -------
        StepFunctionsPipeline
            Self, for method chaining.

        Raises
        ------
        ValueError
            If a step with the same name already exists.
        """
        existing = {s.name for s in self._steps}
        if step.name in existing:
            raise ValueError(f"A step named '{step.name}' already exists in this pipeline")
        self._steps.append(step)
        return self

    def definition(self) -> dict[str, Any]:
        """
        Build the Amazon States Language (ASL) state machine definition.

        Returns
        -------
        dict[str, Any]
            ASL definition ready to pass to the Step Functions API or
            serialize to JSON.

        Raises
        ------
        ValueError
            If no steps have been added.
        """
        if not self._steps:
            raise ValueError("Pipeline has no steps — call add_step() first")

        states: dict[str, Any] = {}
        for i, step in enumerate(self._steps):
            next_state = self._steps[i + 1].name if i + 1 < len(self._steps) else None
            states[step.name] = step.to_asl_state(next_state)

        defn: dict[str, Any] = {
            "Comment": self.comment or f"{self.name} — built by clinops.orchestrate",
            "StartAt": self._steps[0].name,
            "States": states,
        }
        return defn

    def definition_json(self, indent: int = 2) -> str:
        """
        Return the ASL definition as a JSON string.

        Parameters
        ----------
        indent:
            JSON indentation. Default 2.

        Returns
        -------
        str
        """
        return json.dumps(self.definition(), indent=indent)

    def deploy(self) -> str:
        """
        Create or update the state machine in AWS.

        If a state machine with ``self.name`` already exists in the account,
        its definition is updated in-place. Otherwise a new state machine is
        created.

        Returns
        -------
        str
            ARN of the deployed state machine.

        Raises
        ------
        ImportError
            If ``boto3`` is not installed.
        """
        client = self._get_client()
        defn_json = self.definition_json()

        # Check if it already exists
        existing_arn: str | None = self._find_existing_arn(client)

        if existing_arn:
            client.update_state_machine(
                stateMachineArn=existing_arn,
                definition=defn_json,
                roleArn=self.role_arn,
            )
            logger.info(f"StepFunctionsPipeline: updated '{self.name}' → {existing_arn}")
            return existing_arn
        else:
            response = client.create_state_machine(
                name=self.name,
                definition=defn_json,
                roleArn=self.role_arn,
                type="STANDARD",
            )
            arn: str = response["stateMachineArn"]
            logger.info(f"StepFunctionsPipeline: created '{self.name}' → {arn}")
            return arn

    def execute(
        self,
        input_data: dict[str, Any] | None = None,
        execution_name: str | None = None,
        state_machine_arn: str | None = None,
    ) -> str:
        """
        Start an execution of the deployed state machine.

        Parameters
        ----------
        input_data:
            JSON-serializable dict passed as the execution's input.
        execution_name:
            Optional execution name. Defaults to a timestamp-based name.
        state_machine_arn:
            ARN of the state machine to execute. If None, looks up the
            ARN by ``self.name``.

        Returns
        -------
        str
            ARN of the started execution.

        Raises
        ------
        RuntimeError
            If the state machine cannot be found and ``state_machine_arn``
            is not provided.
        """
        client = self._get_client()

        arn = state_machine_arn or self._find_existing_arn(client)
        if not arn:
            raise RuntimeError(
                f"State machine '{self.name}' not found — call deploy() first "
                "or pass state_machine_arn explicitly"
            )

        exec_name = execution_name or f"{self.name}-{int(time.time())}"
        response = client.start_execution(
            stateMachineArn=arn,
            name=exec_name,
            input=json.dumps(input_data or {}),
        )
        execution_arn: str = response["executionArn"]
        logger.info(f"StepFunctionsPipeline: started execution '{exec_name}' → {execution_arn}")
        return execution_arn

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "Step Functions support requires the 'aws' extra: pip install clinops[aws]"
                ) from exc

            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            self._client = session.client("stepfunctions")

        return self._client

    def _find_existing_arn(self, client: Any) -> str | None:
        paginator = client.get_paginator("list_state_machines")
        for page in paginator.paginate():
            for sm in page.get("stateMachines", []):
                if sm["name"] == self.name:
                    return str(sm["stateMachineArn"])
        return None
