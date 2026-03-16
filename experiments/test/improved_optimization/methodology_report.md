# Improved Optimization Methodology Report

## Overview and motivation

The legacy IEEE-30 surrogate workflow exposed a wildfire-aware objective but used broad control spaces and often stalled immediately at baseline. The improved workflow narrows the control set to buses that show line-relief leverage against wildfire-sensitive lines.

## Legacy methodology and observed issue

The legacy workflow optimized over all PV redispatch variables and all PQ shedding variables from a baseline initial point using `L-BFGS-B`. In practice the surrogate frequently exposed a flat local landscape and the optimizer returned zero iterations.

## Improved methodology

The improved workflow adds a screening stage:

- identify wildfire-sensitive lines
- compute finite-difference relief scores for candidate PV generators
- compute finite-difference relief scores for candidate PQ loads
- select the top scoring controls
- optimize over the compact control vector only

## Mathematical formulation

Decision vector:

$u = [\Delta P_g^{selected}, s^{targeted}]$

Objective:

$J(u) = \lambda_g C_gen(u) + \lambda_s C_shed(u) + \lambda_w C_wf(u)$

Generator deviation cost:

$C_gen(u) = \sum_g a_g (\Delta P_g)^2$

Load shedding cost:

$C_shed(u) = \sum_i c_i s_i$

Wildfire penalty:

$C_wf(u) = \sum_{\ell \in L_r} w_\ell \phi(\rho_\ell(u))$

Finite-difference relief score:

$Score_g = \sum_{\ell \in L_r} w_\ell max(0, -S_{g,\ell})$

$Score_d = \sum_{\ell \in L_r} w_\ell max(0, -S_{d,\ell})$

## Decision-variable selection procedure

The current implementation uses surrogate finite differences around baseline with configurable perturbation `epsilon_fd`. Wildfire-sensitive lines are identified from baseline loading using the wildfire threshold minus a buffer, with a top-N fallback when too few lines exceed the buffered threshold.

## Wildfire penalty construction

The improved workflow supports a smooth softplus-squared penalty and a thresholded quadratic fallback. The default is softplus with configurable threshold and sharpness.

## Optimization algorithm

The compact targeted decision vector is optimized with `scipy.optimize.minimize(method='L-BFGS-B')` under simple box constraints for redispatch and shedding.

## Experiment setup

The implementation reuses:

- IEEE-30 test scenario loading
- surrogate model wrappers
- GNN and GPS checkpoint loading
- simplified branch-loading reconstruction from surrogate voltages

## Baseline vs improved results

Automated run summaries are appended by the scripts below this section.

## Pareto analysis

The Pareto sweep varies the shedding penalty while holding the wildfire and generator weights fixed. Outputs include wildfire-vs-shed and objective-vs-wildfire frontiers.

## Objective sensitivity analysis

The sensitivity script evaluates:

- shedding weight sensitivity
- wildfire threshold sensitivity
- softplus sharpness sensitivity
- local one-dimensional objective slices around baseline

## Practical interpretation of load shedding as wildfire mitigation

The improved methodology is designed to answer whether targeted shedding can become an intentional wildfire-mitigation tool once the decision space is restricted to influential buses.

## Limitations

- the surrogate remains the dominant modeling limitation
- branch flow reconstruction is still simplified
- the control-selection scores are local finite-difference approximations
- the current implementation still depends on the restored homogeneous IEEE-30 data path

## Next steps

- compare selected-control sensitivities against a power-flow reference
- calibrate line weights with true wildfire exposure inputs
- revisit optimizer choice after confirming the improved objective landscape is informative
- extend the compact decision vector to future topology controls if needed
## GNN improved optimization run

- `model_name`: gnn
- `success`: True
- `message`: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
- `n_iter`: 2
- `baseline_objective`: 15452021.387699107
- `optimized_objective`: 15451948.03527969
- `baseline_wildfire_cost`: 1545202.1387699107
- `optimized_wildfire_cost`: 1545193.3365464988
- `optimized_total_shed_mw`: 0.29339629399766476
- `selected_generator_buses`: []
- `selected_load_buses`: [2, 18, 16, 23, 25]
- `validation_all_passed`: True
- `history_csv`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gnn_improved_20260316_125940\objective_history.csv
- `run_dir`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gnn_improved_20260316_125940
## GPS improved optimization run

- `model_name`: gps
- `success`: True
- `message`: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- `n_iter`: 3
- `baseline_objective`: 5850904.845002404
- `optimized_objective`: 5836944.859409413
- `baseline_wildfire_cost`: 585090.4845002404
- `optimized_wildfire_cost`: 583660.5689302954
- `optimized_total_shed_mw`: 6.783402129188596
- `selected_generator_buses`: []
- `selected_load_buses`: [25, 18, 23, 17, 2]
- `validation_all_passed`: True
- `history_csv`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gps_improved_20260316_130002\objective_history.csv
- `run_dir`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gps_improved_20260316_130002
## GNN improved optimization run

- `model_name`: gnn
- `success`: False
- `message`: ABNORMAL: 
- `n_iter`: 0
- `baseline_objective`: 15452436.168815361
- `optimized_objective`: 15452436.168815361
- `baseline_wildfire_cost`: 1545243.616881536
- `optimized_wildfire_cost`: 1545243.616881536
- `optimized_total_shed_mw`: 0.0
- `selected_generator_buses`: [1, 4, 7, 10, 12]
- `selected_load_buses`: [2, 18, 16, 23, 25]
- `validation_all_passed`: True
- `history_csv`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gnn_improved_20260316_130227\objective_history.csv
- `run_dir`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gnn_improved_20260316_130227
## GPS improved optimization run

- `model_name`: gps
- `success`: True
- `message`: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
- `n_iter`: 3
- `baseline_objective`: 6112572.724357759
- `optimized_objective`: 6100904.061505817
- `baseline_wildfire_cost`: 611257.2724357758
- `optimized_wildfire_cost`: 610055.3632668165
- `optimized_total_shed_mw`: 7.008576753055646
- `selected_generator_buses`: [1, 4, 7, 10, 12]
- `selected_load_buses`: [25, 18, 23, 17, 2]
- `validation_all_passed`: True
- `history_csv`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gps_improved_20260316_130254\objective_history.csv
- `run_dir`: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\improved_optimization\results\improved_runs\gps_improved_20260316_130254
