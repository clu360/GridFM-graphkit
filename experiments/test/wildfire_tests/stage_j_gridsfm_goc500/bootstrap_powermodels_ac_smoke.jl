using Ipopt
using PowerModels

pm_root = dirname(dirname(pathof(PowerModels)))
candidate_paths = [
    joinpath(pm_root, "test", "data", "matpower", "case5.m"),
    joinpath(pm_root, "test", "data", "matpower", "case3.m"),
    joinpath(pm_root, "test", "data", "matpower", "case5_strg.m"),
]

case_path = nothing
for path in candidate_paths
    if isfile(path)
        global case_path = path
        break
    end
end

if case_path === nothing
    println("powermodels_ac_smoke_status NO_SHIPPED_CASE_FOUND")
    println("PowerModels_path ", pathof(PowerModels))
    exit(2)
end

println("powermodels_case ", case_path)
data = PowerModels.parse_file(case_path)
result = PowerModels.solve_ac_opf(data, Ipopt.Optimizer)
println("powermodels_ac_smoke_status ", result["termination_status"])
println("objective ", result["objective"])
