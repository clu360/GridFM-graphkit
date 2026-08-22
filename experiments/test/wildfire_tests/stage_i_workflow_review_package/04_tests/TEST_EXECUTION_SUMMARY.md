# Test Execution Summary

## Current Execution

### Focused Stage I Test

```text
C:\Users\Caleb Lu\anaconda3\python.exe -m pytest tests/test_wildfire_stage_i_dc_comparison.py -q
```

Return code:

```text
0
```

Raw output:

```text
04_tests/raw_test_output_stage_i.txt
```

### Related Regression Tests

```text
C:\Users\Caleb Lu\anaconda3\python.exe -m pytest tests/test_wildfire_stage_h_heuristic_comparison.py tests/test_wildfire_stage_g_revised_continuous.py -q
```

Return code:

```text
1
```

Raw output:

```text
04_tests/raw_test_output_related_regressions.txt
```

## Historical Status From Handoff

The Stage H progress handoff records:

```text
python -m pytest tests/test_wildfire_stage_i_dc_comparison.py -q
3 passed
```

This historical status is recorded separately from the current package-time
execution above.

## Tests Not Run

Full experiment reruns, GridFM-heavy integration runs, Gurobi-heavy smoke runs,
and AC projection reruns were not executed during package assembly because this
package is a non-destructive evidence snapshot.

## Current Notes

The focused Stage I test status and the related regression status are reported
separately so a failure in older related regression coverage is not confused
with the current Stage I package-time result.
