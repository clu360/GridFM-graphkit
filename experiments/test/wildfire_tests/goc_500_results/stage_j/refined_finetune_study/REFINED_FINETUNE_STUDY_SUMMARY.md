# Stage J Refined Fine-Tuning Study Summary

## Status

```text
FT7_P0_PREFLIGHT_PASS
FT7_P1A_M2_OPS_PASS
FT7_P1B_M3_OPS_PASS
FT7_P2_EXACT_AND_CORE_PASS
FT7_P3_ALL_CANDIDATE_PARETO_PASS
FT7_P4_WARM_START_PASS
FT7_P4_SUMMARY_PASS
```

The study is complete through semantic validation and figure generation. It
compares Guided-DC and four GridSFM checkpoints without TH:

1. released GridSFM v1.1 (M0);
2. FullTop-1000 fine-tuned GridSFM (M1);
3. sequential FullTop-1500 GridSFM (M2);
4. sequential FullTop-1000 plus N-1-500 GridSFM (M3).

M2 and M3 were the only new OPS executions. DC, M0, and M1 are reused,
hash-bound evidence. The Stage J scenarios, five lambda values, topology and
continuous budgets, selected-load correction, PAC terms, and exact Reference
A/B formulations were unchanged. The new guided-only runs completed 15
settings, 1,500 topology rows, 30,000 candidate evaluations, 15 finalists, 15
Reference A solves, 30 Reference B solves, and 120 fidelity rows per model.

## Primary Results

Across the 15 selected finalists, mean native `J_trade` was 0.33363 for M0,
0.30843 for M1, 0.30351 for M2, and 0.30979 for M3. Relative to M0, M2 reduced
this diagnostic objective by about 9.0% and M3 by about 7.1%. M2 also improved
on M1 by about 1.6%; M3 was about 0.4% higher than M1. Guided-DC had the lowest
native mean `J_trade` at 0.26047, but it is an approximation-backed state and
does not provide the full state families used for the GridSFM fidelity audit.

Reference A reinforces the fine-tuning result. Mean exact AC cost was
453,440.84 for M0, 454,311.95 for M1, 451,374.56 for M2, and 452,511.89 for
M3. M2 had the smallest mean absolute native-to-exact risk discrepancy
(0.01762) and `J_trade` discrepancy (0.00511); M3 was second (0.02182 and
0.00704). Across the seven comparable GridSFM state families, mean NRMSE was
0.20726 for M0, 0.14772 for M1, 0.12059 for M2, and 0.14150 for M3. M2 was best
on this FullTop OPS diagnostic. M3 still improved substantially over M0 and
generally over M1, while its FT6 held-out result remains the stronger N-1
composite. This is consistent with distribution-specific continuation rather
than a claim that one child checkpoint dominates in every setting.

## Pareto Study

The risk/load Pareto analysis uses every eligible finite candidate evaluation,
pooled across all five lambda values within each scenario/model group. It does
not use only the final topology or one finalist per lambda. Of 150,000 total
candidate rows, 149,601 met the predeclared eligibility rule (`ok` or
`model_output_penalized` with finite risk and load values). Recomputed
nondominance produced 603 points across all 15 required scenario/model fronts:
111 DC, 138 M0, 81 M1, 93 M2, and 180 M3 points. Frontier size is descriptive
of sampled candidate diversity and is not itself a quality score.

## Warm-Start Study

The crossed timing study contains exactly 525 rows:

```text
5 fixed finalist families x 15 settings x 7 initialization policies
```

The starts are cold, DC partial, full M0, full M1, full M2, full M3, and exact
Reference A primal. The final timing evidence comes from a controlled rerun of
all 525 solves, not the earlier mixed-batch aggregate. Start order was rotated
across the seven execution positions, with each start appearing 10 or 11 times
in every position. The primary metric is solver-reported IPOPT solve time;
model construction, start loading, GridSFM inference, and result export remain
excluded.

Every solve succeeded on the same fixed Reference A instance, all full GridSFM
starts supplied `Pg,Qg,V,theta`, and every solve reported a positive IPOPT
barrier-iteration count. The maximum objective spread across starts was
`8.41e-06`, below the `1e-3` tolerance. The cold policy was explicitly verified
as `V=1, theta=0, Pg=(Pmin+Pmax)/2, Qg=0`.

Mean and median IPOPT results were:

| Initialization | Mean time (s) | Median time (s) | Time wins vs cold | Mean iterations | Median iterations | Iteration wins vs cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 5.221 | 5.113 | reference | 33.64 | 33 | reference |
| DC | 5.151 | 5.075 | 37/75 | 31.57 | 30 | 64/75 |
| M0 full state | 5.160 | 5.104 | 39/75 | 32.25 | 31 | 50/75 |
| M1 full state | 5.108 | 5.039 | 48/75 | 32.45 | 31 | 51/75 |
| M2 full state | 5.104 | 5.055 | 45/75 | 32.49 | 31 | 50/75 |
| M3 full state | 5.159 | 5.067 | 41/75 | 32.32 | 31 | 51/75 |
| Exact A primal | 5.216 | 5.089 | 42/75 | 32.43 | 31 | 44/75 |

The controlled result removes the earlier apparent multi-second M2/M3
slowdown, confirming that it was a mixed-batch artifact. M1 and M2 are
effectively tied in mean IPOPT time: M2 is lower by only 0.004 seconds, while
M1 has the lower median and more paired time wins. All four GridSFM starts
reduce the median iteration count from 33 to 31, but the checkpoints have very
similar iteration distributions. DC has the lowest median iteration count at
30, although this does not translate into the lowest mean solve time. Iteration
count and solve time are related but not interchangeable because IPOPT
iterations can differ in computational cost. The evidence supports modest
warm-start improvements, especially for M1/M2, but not a strong checkpoint
ordering by solver speed.

## Claim Boundary

The results support the claim that official GridSFM/OPFData fine-tuning can be
applied directly to the modified Stage J OPS workflow and can improve native
and exact-projected diagnostic quality. They do not establish universal
economic superiority, security-constrained performance, or universal solver
speedup.
The DC and GridSFM native outputs have different state completeness, the load
shedding range remains small in this diagnostic setup, and M3's principal
benefit is topology-distribution exposure rather than dominance on the
FullTop-oriented OPS study.
