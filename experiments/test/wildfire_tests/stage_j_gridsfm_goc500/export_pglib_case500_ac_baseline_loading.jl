using Ipopt
using PowerModels

if length(ARGS) < 3
    println("Usage: julia export_pglib_case500_ac_baseline_loading.jl <pglib_case.m> <output_csv> <summary_json>")
    exit(2)
end

case_path = ARGS[1]
output_csv = ARGS[2]
summary_json = ARGS[3]

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

mkpath(dirname(output_csv))
mkpath(dirname(summary_json))

function as_float(value)
    return Float64(value)
end

rows = Vector{Tuple{Int,Int,Int,Float64,Float64,Float64,Float64,Float64,Float64,Int}}()

if haskey(result, "solution") && haskey(result["solution"], "branch")
    solution_branches = result["solution"]["branch"]
    for (branch_id_text, branch) in sort(collect(data["branch"]); by = x -> parse(Int, x[1]))
        branch_id = parse(Int, branch_id_text)
        status_value = Int(get(branch, "br_status", 1))
        if status_value != 1
            continue
        end
        rate_a = as_float(get(branch, "rate_a", 0.0))
        if !isfinite(rate_a) || rate_a <= 0.0
            continue
        end
        sol = solution_branches[string(branch_id)]
        pf = as_float(get(sol, "pf", 0.0))
        qf = as_float(get(sol, "qf", 0.0))
        pt = as_float(get(sol, "pt", 0.0))
        qt = as_float(get(sol, "qt", 0.0))
        s_from = sqrt(pf^2 + qf^2)
        s_to = sqrt(pt^2 + qt^2)
        loading = max(s_from, s_to) / rate_a
        push!(
            rows,
            (
                branch_id,
                Int(branch["f_bus"]),
                Int(branch["t_bus"]),
                rate_a,
                pf,
                qf,
                pt,
                qt,
                loading,
                status_value,
            ),
        )
    end
end

open(output_csv, "w") do io
    println(io, "branch_id,f_bus,t_bus,rate_a,pf,qf,pt,qt,baseline_loading,br_status")
    for row in rows
        println(io, join(row, ","))
    end
end

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa AbstractString
        return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Number
        return string(value)
    else
        return "\"" * string(value) * "\""
    end
end

positive_rate_count = sum(row[4] > 0.0 && isfinite(row[4]) for row in rows)
max_loading = isempty(rows) ? nothing : maximum(row[9] for row in rows)
overloaded_count = sum(row[9] > 1.0 for row in rows)
pd_total = sum(as_float(load["pd"]) for (_, load) in data["load"])
qd_total = sum(as_float(load["qd"]) for (_, load) in data["load"])

summary = Dict(
    "case_path" => case_path,
    "termination_status" => status,
    "objective" => objective,
    "runtime_seconds" => runtime,
    "baseMVA" => data["baseMVA"],
    "bus_count" => length(data["bus"]),
    "load_count" => length(data["load"]),
    "generator_count" => length(data["gen"]),
    "branch_count" => length(data["branch"]),
    "exported_active_branch_count" => length(rows),
    "positive_rate_exported_branch_count" => positive_rate_count,
    "max_baseline_loading" => max_loading,
    "num_baseline_loading_gt_1" => overloaded_count,
    "pd_total" => pd_total,
    "qd_total" => qd_total,
    "loading_definition" => "max(sqrt(pf^2+qf^2),sqrt(pt^2+qt^2))/rate_a",
)

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
    "exported_active_branch_count",
    "positive_rate_exported_branch_count",
    "max_baseline_loading",
    "num_baseline_loading_gt_1",
    "pd_total",
    "qd_total",
    "loading_definition",
]

open(summary_json, "w") do io
    println(io, "{")
    for (idx, key) in enumerate(json_keys)
        comma = idx == length(json_keys) ? "" : ","
        println(io, "  \"", key, "\": ", json_value(summary[key]), comma)
    end
    println(io, "}")
end

println("baseline_loading_status ", status)
println("exported_active_branch_count ", length(rows))
println("max_baseline_loading ", max_loading)
