using Ipopt
using PowerModels

if length(ARGS) < 2
    println("Usage: julia bootstrap_pglib_case500_ac_baseline.jl <pglib_case.m> <output_json>")
    exit(2)
end

case_path = ARGS[1]
output_path = ARGS[2]

PowerModels.silence()

data = PowerModels.parse_file(case_path; validate=true)

solver = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "max_iter" => 10000,
)

t0 = time()
result = PowerModels.solve_ac_opf(data, solver)
runtime = time() - t0

status = string(result["termination_status"])
objective = haskey(result, "objective") ? result["objective"] : nothing

loads = data["load"]
branches = data["branch"]
gens = data["gen"]

pd_total = sum(Float64(load["pd"]) for (_, load) in loads)
qd_total = sum(Float64(load["qd"]) for (_, load) in loads)
active_branches = sum(Int(get(branch, "br_status", 1)) == 1 for (_, branch) in branches)

summary = Dict(
    "case_path" => case_path,
    "termination_status" => status,
    "objective" => objective,
    "runtime_seconds" => runtime,
    "baseMVA" => data["baseMVA"],
    "bus_count" => length(data["bus"]),
    "load_count" => length(loads),
    "generator_count" => length(gens),
    "branch_count" => length(branches),
    "active_branch_count" => active_branches,
    "pd_total" => pd_total,
    "qd_total" => qd_total,
)

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa AbstractString
        return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    elseif value isa Number
        return string(value)
    else
        return "\"" * string(value) * "\""
    end
end

json_keys = [
    "case_path",
    "termination_status",
    "objective",
    "runtime_seconds",
    "baseMVA",
    "bus_count",
    "load_count",
    "generator_count",
    "branch_count",
    "active_branch_count",
    "pd_total",
    "qd_total",
]

open(output_path, "w") do io
    println(io, "{")
    for (idx, key) in enumerate(json_keys)
        comma = idx == length(json_keys) ? "" : ","
        println(io, "  \"", key, "\": ", json_value(summary[key]), comma)
    end
    println(io, "}")
end

println("pglib_case500_ac_baseline_status ", status)
println("objective ", objective)
println("runtime_seconds ", runtime)
