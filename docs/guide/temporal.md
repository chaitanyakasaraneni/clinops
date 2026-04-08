# Temporal

`clinops.temporal` provides time-series windowing, gap-aware imputation, lag/rolling features, and cohort alignment for ICU and clinical event data.

---

## TemporalWindower

Extracts sliding or tumbling feature windows from a longitudinal DataFrame.

```python
from clinops.temporal import TemporalWindower

windower = TemporalWindower(window_hours=24, step_hours=6)

windows = windower.fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    feature_cols=["heart_rate", "spo2", "resp_rate", "map"],
)
# Returns: subject_id | window_start | window_end | heart_rate | spo2 | ...
```

### Long-format input (MIMIC native)

MIMIC chartevents stores data in long format (`itemid × valuenum`). Pass `item_col` and `value_col` to auto-pivot:

```python
windows = windower.fit_transform(
    df=charts,
    id_col="subject_id",
    time_col="charttime",
    item_col="itemid",
    value_col="valuenum",
)
```

### Minimum observations

Skip sparse windows with fewer than `min_observations` non-null measurements:

```python
windower = TemporalWindower(
    window_hours=24,
    step_hours=6,
    min_observations=3,
)
```

---

## Imputer

Gap-aware imputation for clinical time-series. When `id_col` is provided, fill is applied within each patient/admission independently — values never propagate across entity boundaries.

```python
from clinops.temporal import Imputer, ImputationStrategy

imputer = Imputer(
    ImputationStrategy.FORWARD_FILL,
    max_gap_hours=6,       # re-null filled values spanning gaps > 6 hours
    time_col="charttime",
    id_col="subject_id",
)
imputer.fit(train_windows)
test_windows = imputer.transform(test_windows)
```

### Strategies

| Strategy | Best for |
|---|---|
| `FORWARD_FILL` | Slowly-changing vitals — carry last observation forward |
| `BACKWARD_FILL` | Values recorded with a documentation lag |
| `LINEAR` | Continuous signals with regular sampling |
| `MEAN` | Fit on training set, fill with per-column or per-patient mean |
| `MEDIAN` | As above, more robust to skewed lab distributions |
| `INDICATOR` | Adds `{col}_missing` binary column — lets the model learn from missingness |
| `NONE` | Leave `NaN` in place |

### Per-patient mean/median

Compute imputation statistics per patient rather than globally (avoids leakage when patients have very different baseline values):

```python
imputer = Imputer(
    ImputationStrategy.MEAN,
    per_patient=True,
    id_col="subject_id",
)
imputer.fit(train_windows)
```

### max_gap_hours

For `FORWARD_FILL` and `BACKWARD_FILL`, `max_gap_hours` prevents propagating stale values across long gaps. A NaN that was filled across a gap larger than the threshold is restored to `NaN`:

```python
# Fill gaps up to 4 hours; leave longer gaps as NaN
imputer = Imputer(
    ImputationStrategy.FORWARD_FILL,
    max_gap_hours=4,
    time_col="charttime",
    id_col="subject_id",
)
```

---

## LagFeatureBuilder

Add lag columns and rolling statistics to a windowed DataFrame.

```python
from clinops.temporal import LagFeatureBuilder

enriched = LagFeatureBuilder(
    lags=[1, 2, 4],
    rolling_windows=[4, 8],
    id_col="subject_id",
).fit_transform(windows)
# Adds: heart_rate_lag1, heart_rate_lag2, heart_rate_lag4,
#        heart_rate_roll4_mean, heart_rate_roll4_std,
#        heart_rate_roll8_mean, heart_rate_roll8_std, ...
```

Lags and rolling statistics are computed per-patient when `id_col` is set, preventing values from one patient's last window from appearing as features in the next patient's first window.

---

## CohortAligner

Align a cohort's event time-series to a common reference event (e.g. ICU admission time), then filter to a time window around that anchor.

```python
from clinops.temporal import CohortAligner

aligned = CohortAligner(
    anchor_col="intime",
    max_hours_before=0,
    max_hours_after=48,
).align(events_df=charts, anchor_df=stays)
# Returns charts filtered to 48 hours post-admission,
# with an additional hours_from_anchor column
```

Useful for building mortality-prediction cohorts (48-hour ICU window), early-warning datasets, and any study with a defined index event.
