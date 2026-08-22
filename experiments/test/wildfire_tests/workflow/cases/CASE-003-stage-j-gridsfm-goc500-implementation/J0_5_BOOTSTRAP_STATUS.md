# J0.5 Environment / Model / Data Bootstrap Status v001

```yaml
artifact_id: CASE-003-J0.5-BOOTSTRAP-STATUS
artifact_version: v001
created_local: 2026-08-12T22:58:28.1336018-04:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: BOOTSTRAP_PASS_WITH_NEXT_GATES
sha256_when_frozen: null
```

## Summary

J0.5 is complete for the external GridSFM model environment, official checkpoint, official GOC-500 sample loading/preprocessing, official inference smoke, and Julia/PowerModels/IPOPT solver smoke.

This does not yet authorize the full S1-S3 wildfire experiment. The next gates are the GOC-500 branch identity adapter, candidate raw-mutation pipeline, fixed-topology economic DC-OPF recourse, intact GOC-500 economic AC baseline, and PAC calibration.

## Bootstrap Results

| Gate | Status | Evidence |
|---|---:|---|
| Locate or obtain GridSFM repo | PASS | Official Microsoft repo cloned to `C:/Users/Caleb Lu/.gridfm_stage_j/repos/GridSFM` at commit `1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`. |
| Read official setup docs | PASS | Read top-level `README.md`, `model/README.md`, `model/samples/README.md`, `model/examples/infer_samples.py`, and `model/pyproject.toml`. |
| Isolated Python environment | PASS | Created `C:/Users/Caleb Lu/.gridfm_stage_j/envs/gridsfm`; installed `pip install -e "model[test]"`. |
| Import-only smoke | PASS | Python 3.12.11, `torch 2.8.0+cpu`, `torch_geometric 2.8.0.post1`, `huggingface_hub 0.36.2`, `gridsfm 1.1.0`. |
| Official checkpoint download | PASS | `microsoft/GridSFM_Open`, revision `1b41299b80252adf1869d5c3b479a4a402c52591`, file `gridsfm_open_v1.1.pt`, SHA-256 `F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831`. |
| Checkpoint load | PASS | Loaded as `GridTransformerBackbone` on CPU. |
| Required GOC-500 data load | PASS | Official shipped sample `case500_goc.pyg.json`, SHA-256 `66457EF63E2F496757094D06DE0EC3CAFB59561CE074094828041E549ED53E5C`. |
| Official preprocessing | PASS | `load_pyg_json -> prepare_for_inference`; raw bus features `(500,4)` became prepared `(500,16)` with `branch_ac`, `branch_tr`, and `cycle` node types. |
| Single-case inference | PASS | `predict(model, case500_goc)` returned `V/theta/Pg/Qg/Pij/Qij/Pji/Qji`, flow types `['ac_line','transformer']`, counts `[536,192]`, `feas=0.9973402619361877`. |
| Official example inference | PASS | `model/examples/infer_samples.py` ran all 53 shipped samples on CPU; feasibility accuracy `53/53`, mean cost MAPE `0.61%`, prep `15.2s`, forward `72.3s`. |
| Julia install | PASS | Official portable Julia 1.10.11 installed under `C:/Users/Caleb Lu/.gridfm_stage_j/tools/`; ZIP SHA-256 `11BA52FD1384F82D09EA232EB1552B6694BB2083E6ADFE3AE2F9E1E663ED8CF8`. |
| JuMP/PowerModels/IPOPT install | PASS | `JuMP 1.31.1`, `Ipopt 1.15.0`, `PowerModels 0.21.6` installed in external Julia depot. |
| Trivial Ipopt solve | PASS | JuMP scalar quadratic returned `LOCALLY_SOLVED`, objective `0.0`. |
| PowerModels AC-OPF smoke | PASS | PowerModels shipped `case5.m` solved with Ipopt, termination `LOCALLY_SOLVED`, objective `18269.102722788884`. |

## Important Limitations

The official GridSFM README states the model is tested on Ubuntu/macOS, while this bootstrap ran on Windows CPU. The import, checkpoint load, preprocessing, and inference path all succeeded anyway, but Windows remains a platform difference to keep in the environment manifest.

The official shipped `case500_goc.pyg.json` verifies the GOC-500-compatible native schema and inference path. The Stage J experimental adapter still needs to mutate raw topology/load and then call official preprocessing for every candidate `(z, alpha)`.

The exact AC smoke only proves Julia/PowerModels/IPOPT can solve a shipped small AC-OPF case. It does not yet prove the final GOC-500 fixed-`z,alpha` economic AC-OPF and Rhodes-style MLD audits are implemented.

## Next Required Gates

1. J2: immutable canonical branch identity table for GOC-500.
2. J3: raw topology/load mutation plus official GridSFM preprocessing for candidate requests.
3. Guided-DC: fixed-`(z,alpha)` economic DC-OPF recourse implementation.
4. J4: intact economic AC-OPF baseline for admitted electrical scenarios.
5. J5: PAC scaling calibration on intact plus selected N-1/N-2 smoke states.
