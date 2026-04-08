"""
Cloud storage for clinops pipeline artifacts.

Provides a consistent upload/download interface for persisting DataFrames
and arbitrary files to GCS or S3 between pipeline steps. Designed for
scheduled clinical data engineering workflows where intermediate results
need to be checkpointed across runs or shared between services.

Both stores use Parquet by default for efficient storage of wide clinical
DataFrames. CSV is also supported for interoperability with downstream
tools that cannot read Parquet.

Optional dependencies
---------------------
GCS requires ``pip install clinops[gcp]`` (google-cloud-storage).
S3 requires ``pip install clinops[aws]`` (boto3).
"""

from __future__ import annotations

import io
from enum import StrEnum
from typing import Any

import pandas as pd
from loguru import logger


class StorageFormat(StrEnum):
    """Serialization format for pipeline artifacts."""

    PARQUET = "parquet"
    CSV = "csv"


class GCSPipelineStore:
    """
    Upload and download clinops pipeline artifacts to Google Cloud Storage.

    Requires ``pip install clinops[gcp]``.

    Parameters
    ----------
    bucket:
        GCS bucket name (without ``gs://`` prefix).
    prefix:
        Optional path prefix applied to all artifact names. Use to
        namespace artifacts by pipeline or environment, e.g.
        ``"clinops/prod/v2"``.
    format:
        Serialization format. Default ``StorageFormat.PARQUET``.
    credentials_path:
        Path to a service account JSON key file. If None, falls back to
        Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``
        env var or gcloud CLI credentials).

    Examples
    --------
    >>> store = GCSPipelineStore("my-clinical-bucket", prefix="clinops/prod")
    >>> store.upload(windows_df, "features/windows_2024_01")
    'gs://my-clinical-bucket/clinops/prod/features/windows_2024_01.parquet'

    >>> df = store.download("features/windows_2024_01")
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        format: StorageFormat = StorageFormat.PARQUET,
        credentials_path: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.format = format
        self.credentials_path = credentials_path
        self._client: Any = None

    def upload(self, df: pd.DataFrame, name: str) -> str:
        """
        Serialize ``df`` and upload it to GCS.

        Parameters
        ----------
        df:
            DataFrame to upload.
        name:
            Artifact name (without extension). Forward slashes create
            virtual directories, e.g. ``"features/windows_2024_01"``.

        Returns
        -------
        str
            Full GCS URI of the uploaded object (``gs://bucket/path``).
        """
        client = self._get_client()
        blob_path = self._blob_path(name)
        buffer = self._serialize(df)

        bucket_obj = client.bucket(self.bucket)
        blob = bucket_obj.blob(blob_path)
        blob.upload_from_file(buffer, content_type=self._content_type())

        uri = f"gs://{self.bucket}/{blob_path}"
        logger.info(f"GCSPipelineStore: uploaded {len(df):,} rows → {uri}")
        return uri

    def download(self, name: str) -> pd.DataFrame:
        """
        Download and deserialize a DataFrame artifact from GCS.

        Parameters
        ----------
        name:
            Artifact name (without extension), matching what was passed to
            ``upload()``.

        Returns
        -------
        pd.DataFrame
        """
        client = self._get_client()
        blob_path = self._blob_path(name)

        bucket_obj = client.bucket(self.bucket)
        blob = bucket_obj.blob(blob_path)
        data = blob.download_as_bytes()

        df = self._deserialize(data)
        logger.info(
            f"GCSPipelineStore: downloaded {len(df):,} rows ← gs://{self.bucket}/{blob_path}"
        )
        return df

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """
        List artifact names stored under an optional sub-prefix.

        Parameters
        ----------
        prefix:
            Sub-prefix to filter by (relative to the store's base prefix).

        Returns
        -------
        list[str]
            Artifact names (without file extension), relative to the store's
            base prefix.
        """
        client = self._get_client()
        base = f"{self.prefix}/{prefix}".strip("/")
        blobs = client.list_blobs(self.bucket, prefix=base)
        ext = f".{self.format}"
        names = []
        for blob in blobs:
            name = blob.name
            if name.endswith(ext):
                rel = name[len(self.prefix) :].lstrip("/").removesuffix(ext)
                names.append(rel)
        return sorted(names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
            except ImportError as exc:
                raise ImportError(
                    "GCS support requires the 'gcp' extra: pip install clinops[gcp]"
                ) from exc

            if self.credentials_path:
                creds = service_account.Credentials.from_service_account_file(  # type: ignore
                    self.credentials_path
                )
                self._client = storage.Client(credentials=creds)
            else:
                self._client = storage.Client()

        return self._client

    def _blob_path(self, name: str) -> str:
        parts = [p for p in [self.prefix, name] if p]
        return "/".join(parts) + f".{self.format}"

    def _serialize(self, df: pd.DataFrame) -> io.BytesIO:
        buf = io.BytesIO()
        if self.format == StorageFormat.PARQUET:
            df.to_parquet(buf, index=False)
        else:
            df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _deserialize(self, data: bytes) -> pd.DataFrame:
        buf = io.BytesIO(data)
        if self.format == StorageFormat.PARQUET:
            return pd.read_parquet(buf)
        return pd.read_csv(buf)

    def _content_type(self) -> str:
        return "application/octet-stream" if self.format == StorageFormat.PARQUET else "text/csv"


class S3PipelineStore:
    """
    Upload and download clinops pipeline artifacts to Amazon S3.

    Requires ``pip install clinops[aws]``.

    Parameters
    ----------
    bucket:
        S3 bucket name (without ``s3://`` prefix).
    prefix:
        Optional key prefix applied to all artifact names.
    format:
        Serialization format. Default ``StorageFormat.PARQUET``.
    region:
        AWS region name. Default ``"us-east-1"``.
    profile:
        AWS credentials profile name. If None, uses the default credential
        chain (env vars, instance profile, ``~/.aws/credentials``).

    Examples
    --------
    >>> store = S3PipelineStore("my-clinical-bucket", prefix="clinops/prod")
    >>> store.upload(windows_df, "features/windows_2024_01")
    's3://my-clinical-bucket/clinops/prod/features/windows_2024_01.parquet'

    >>> df = store.download("features/windows_2024_01")
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        format: StorageFormat = StorageFormat.PARQUET,
        region: str = "us-east-1",
        profile: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.format = format
        self.region = region
        self.profile = profile
        self._client: Any = None

    def upload(self, df: pd.DataFrame, name: str) -> str:
        """
        Serialize ``df`` and upload it to S3.

        Parameters
        ----------
        df:
            DataFrame to upload.
        name:
            Artifact name (without extension).

        Returns
        -------
        str
            Full S3 URI of the uploaded object (``s3://bucket/key``).
        """
        client = self._get_client()
        key = self._key(name)
        buffer = self._serialize(df)

        client.upload_fileobj(buffer, self.bucket, key)
        uri = f"s3://{self.bucket}/{key}"
        logger.info(f"S3PipelineStore: uploaded {len(df):,} rows → {uri}")
        return uri

    def download(self, name: str) -> pd.DataFrame:
        """
        Download and deserialize a DataFrame artifact from S3.

        Parameters
        ----------
        name:
            Artifact name (without extension).

        Returns
        -------
        pd.DataFrame
        """
        client = self._get_client()
        key = self._key(name)

        buf = io.BytesIO()
        client.download_fileobj(self.bucket, key, buf)
        buf.seek(0)

        df = self._deserialize(buf.read())
        logger.info(f"S3PipelineStore: downloaded {len(df):,} rows ← s3://{self.bucket}/{key}")
        return df

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """
        List artifact names stored under an optional sub-prefix.

        Parameters
        ----------
        prefix:
            Sub-prefix to filter by (relative to the store's base prefix).

        Returns
        -------
        list[str]
            Artifact names (without file extension).
        """
        client = self._get_client()
        base = f"{self.prefix}/{prefix}".strip("/")
        ext = f".{self.format}"
        paginator = client.get_paginator("list_objects_v2")
        names = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=base):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(ext):
                    rel = key[len(self.prefix) :].lstrip("/").removesuffix(ext)
                    names.append(rel)
        return sorted(names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "S3 support requires the 'aws' extra: pip install clinops[aws]"
                ) from exc

            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            self._client = session.client("s3")

        return self._client

    def _key(self, name: str) -> str:
        parts = [p for p in [self.prefix, name] if p]
        return "/".join(parts) + f".{self.format}"

    def _serialize(self, df: pd.DataFrame) -> io.BytesIO:
        buf = io.BytesIO()
        if self.format == StorageFormat.PARQUET:
            df.to_parquet(buf, index=False)
        else:
            df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _deserialize(self, data: bytes) -> pd.DataFrame:
        buf = io.BytesIO(data)
        if self.format == StorageFormat.PARQUET:
            return pd.read_parquet(buf)
        return pd.read_csv(buf)
