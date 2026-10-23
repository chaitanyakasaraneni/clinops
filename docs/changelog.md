# Changelog

## 0.6.1

A patch release with one bug fix and documentation, testing, and packaging
improvements from the [pyOpenSci review](https://github.com/pyOpenSci/software-submission/issues/362).
No other behavior changes.

### Fixed

- `ICDMapper.harmonize(..., output_col=...)` left rows that were already ICD-10
  as NaN in the output column; they are now carried over. Harmonizing in place
  (no `output_col`) was not affected.
  ([#26](https://github.com/chaitanyakasaraneni/clinops/pull/26))

### Documentation

- Tutorials are rendered with outputs in the docs and executed on every build.
  The README links to them. The previous docs links to the notebooks returned 404.
  ([#25](https://github.com/chaitanyakasaraneni/clinops/pull/25))
- Every docstring example is now self-contained and runnable, and CI checks
  that each one produces the output it shows. `Imputer` gained an example, and
  `LagFeatureBuilder`'s example no longer passes `id_col` to `fit_transform()`.
  ([#26](https://github.com/chaitanyakasaraneni/clinops/pull/26))
- Corrected feature counts: `UnitNormalizer` has 26 conversions (not 30) and
  `ClinicalOutlierClipper` has 37 built-in bounds (not 20). The preprocess guide's
  conversion table now matches `UNIT_CONVERSIONS`.
  ([#25](https://github.com/chaitanyakasaraneni/clinops/pull/25))
- Added `CITATION.cff`, which enables GitHub's "Cite this repository".
  ([#25](https://github.com/chaitanyakasaraneni/clinops/pull/25))

### Testing and packaging

- Python 3.13 and 3.14 are tested in CI and listed in the classifiers. Publishing
  to PyPI now requires those jobs and the strict docs build to pass.
  ([#27](https://github.com/chaitanyakasaraneni/clinops/pull/27))
- `clinops.orchestrate` storage and Step Functions I/O are tested against fake
  clients. Total coverage is now 94%.
  ([#27](https://github.com/chaitanyakasaraneni/clinops/pull/27))

## 0.6.0

This release fixes issues found during the
[pyOpenSci review](https://github.com/pyOpenSci/software-submission/issues/362).
Several fixes change results compared with 0.5.0; see **Behavior changes** below.

### Behavior changes

- **eICU labels count only measured values.** `EicuTableLoader` organ labels now
  fire only in hours where the triggering variable was actually measured, instead
  of on the forward-filled grid. Expect fewer positive labels. Set
  `EicuCohortConfig(label_mode="locf")` to reproduce results from 0.5.0 or earlier.
  ([#18](https://github.com/chaitanyakasaraneni/clinops/issues/18))
- **`TemporalWindower(imputation=...)` is applied.** Windows with no observations
  for a feature are now filled across each subject's windows (by default, from the
  previous window). `MEAN` and `MEDIAN` are no longer accepted by the windower;
  fit an `Imputer` on the training split instead.
  ([#15](https://github.com/chaitanyakasaraneni/clinops/issues/15))
- **Stricter schema validation.** `ClinicalSchema.validate` now enforces
  `ColumnSpec.dtype` and `allow_extra_columns=False`, and reports `min_value` or
  `max_value` set on a non-numeric column. Built-in MIMIC schemas are unaffected.
  ([#17](https://github.com/chaitanyakasaraneni/clinops/issues/17))
- **`Imputer` argument checks.** `max_gap_hours` without `time_col`, and
  `per_patient=True` without `id_col`, now raise `ValueError` instead of being
  silently ignored.

### Fixed

- `Imputer(MEAN/MEDIAN, per_patient=True)` now uses the statistics learned in
  `fit()`. Patients unseen at fit time fall back to the fitted global statistic,
  never to statistics of the data being transformed.
  ([#12](https://github.com/chaitanyakasaraneni/clinops/issues/12))
- `max_gap_hours` is measured from each column's last actual observation, so dense
  rows from other features no longer extend a stale value.
  ([#13](https://github.com/chaitanyakasaraneni/clinops/issues/13))
- `Imputer(LINEAR)` interpolates within each `id_col` group.
  ([#14](https://github.com/chaitanyakasaraneni/clinops/issues/14))
- `MimicLoader(chunk_size=...)` no longer raises `TypeError`. MIMIC-IV and MIMIC-III
  loaders apply filters to each chunk, so only matching rows are held in memory.
  ([#16](https://github.com/chaitanyakasaraneni/clinops/issues/16))
- `FHIRLoader.observations(category=...)` now filters by category.
  ([#17](https://github.com/chaitanyakasaraneni/clinops/issues/17))
- pandas 3 compatibility: MIMIC time filters and `UnitNormalizer` with integer
  values. ([#11](https://github.com/chaitanyakasaraneni/clinops/pull/11))

### Documentation and tooling

- README: comparison with PyHealth and scikit-learn, with a combined workflow
  example; module list, extras, and citation updated.
- CI runs the test suite against both pandas 2 and pandas 3, and pins
  `pandas-stubs` for reproducible type checking.

Thanks to [@alex09x](https://github.com/alex09x) for the review issues and fixes.
