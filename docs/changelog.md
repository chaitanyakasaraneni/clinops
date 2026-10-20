# Changelog

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
