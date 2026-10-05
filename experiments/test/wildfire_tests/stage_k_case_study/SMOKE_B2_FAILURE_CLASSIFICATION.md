# Smoke GridSFM Reference B2 Classification

## Classification

The smoke GridSFM Reference B2 outcome is a legitimate nonlinear solver
failure, not an implementation, exception-handling, cost-vector, or export
failure.

Evidence:

- Reference B1 returned `LOCALLY_SOLVED` and certified full maximum service.
- B2 ran for 363 Ipopt iterations and 391.70 solver seconds.
- PowerModels returned `OTHER_ERROR` with an objective and complete exported
  branch, bus, generator, and load-state files.
- Julia produced no exception artifact and the captured stderr was empty.
- The generation-cost diagnostic safely handled missing cost vectors.
- The B1 service target and configured `1e-4 MW` tolerance were passed into the
  unchanged B2 formulation.

Production therefore preserves the frozen B1-to-B2 formulation and cold-start
semantics. If a B2 task is not solver-eligible, B1 remains certified, physical
diagnostics use B1, economic tie-break cost is unset, and the task is retained
as `PASS_WITH_WARNINGS`.
