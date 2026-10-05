# Stage K Cumulative Environmental Hazard And Wildfire Risk

## Scope

This report uses the June 23, 2023 16:00 CDT environmental snapshot for the modified Texas2k grid. The active weather-fused wildfire hazard is the branch-level cumulative FFWI exposure:

```text
hazard_score = p_cumulative
```

The loading-weighted wildfire risk term requested for this package is:

```text
wildfire_risk = p_cumulative * scenario16_baseline_loading
```

We also include the squared-loading wildfire risk term used in the earlier Stage-style risk convention:

```text
wildfire_risk_squared = p_cumulative * scenario16_baseline_loading^2
```

## Summary

- Branches: 5344
- Valid weather coverage branches: 5344
- Hazard median: 0.015342
- Hazard p95: 0.132227
- Hazard max: 1.000000
- Loading-weighted risk median: 0.003396
- Loading-weighted risk p95: 0.053594
- Loading-weighted risk max: 0.523030
- Squared-loading risk median: 0.000762
- Squared-loading risk p95: 0.029748
- Squared-loading risk max: 0.273560

## Outputs

- `figures/pcum_hazard_map.png`
- `figures/pcum_x_loading_map.png`
- `figures/pcum_x_loading2_map.png`
- `figures/pcum_hazard_risk_hist.png`
- `tables/top_pcum_hazard.csv`
- `tables/top_pcum_x_loading.csv`
- `tables/top_pcum_x_loading2.csv`
- `data/cum_hazard_risk.parquet`
- `data/cum_hazard_risk.csv`
- `data/cum_hazard_risk_summary.json`
