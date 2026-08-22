using Ipopt
using PowerModels

if length(ARGS) < 5
    println("Usage: julia reference_a_fixed_alpha_ac_opf.jl <pglib_case.m> <alpha_effective_csv> <offline_branch_ids> <output_summary_json> <output_branch_csv>")
    exit(2)
end

case_path = ARGS[1]
alpha_effective_csv = ARGS[2]
offline_branch_ids_arg = ARGS[3]
output_summary_json = ARGS[4]
output_branch_csv = ARGS[5]

PowerModels.silence()

function parse_offline_branch_ids(value)
    ids = Set{String}()
    for part in split(replace(value, "," => ";"), ";")
        stripped = strip(part)
        if stripped != ""
            push!(ids, string(parse(Int, stripped)))
        end
    end
    return ids
end

function parse_alpha_csv(path)
    alpha = Dict{String, Float64}()
    lines = readlines(path)
    for (idx, line) in enumerate(lines)
        if idx == 1 && occursin("load_id", lowercase(line))
            continue
        end
        parts = split(strip(line), ",")
        if length(parts) >= 2 && strip(parts[1]) != ""
            alpha[string(parse(Int, strip(parts[1])))] = parse(Float64, strip(parts[2]))
        end
    end
    return alpha
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

mkpath(dirname(output_summary_json))
mkpath(dirname(output_branch_csv))

data = PowerModels.parse_file(case_path; validate=true)
alpha = parse_alpha_csv(alpha_effective_csv)
offline_branch_ids = parse_offline_branch_ids(offline_branch_ids_arg)

missing_load_ids = String[]
for (load_id, load) in data["load"]
    if haskey(alpha, string(load_id))
        load["pd"] = Float64(load["pd"]) * alpha[string(load_id)]
        load["qd"] = Float64(load["qd"]) * alpha[string(load_id)]
    else
        push!(missing_load_ids, string(load_id))
    end
end

for branch_id in offline_branch_ids
    if haskey(data["branch"], branch_id)
        data["branch"][branch_id]["br_status"] = 0
    end
end

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
has_solution = haskey(result, "solution") && haskey(result["solution"], "branch")

rows = Vector{Tuple{Int,Int,Int,Float64,Float64,Float64,Float64,Float64,Float64,Int}}()
bus_rows = Vector{Tuple{Int,Float64,Float64}}()
gen_rows = Vector{Tuple{Int,Float64,Float64}}()
if has_solution
    solution_branches = result["solution"]["branch"]
    for (branch_id_text, branch) in sort(collect(data["branch"]); by = x -> parse(Int, x[1]))
        branch_id = parse(Int, branch_id_text)
        status_value = Int(get(branch, "br_status", 1))
        if status_value != 1
            continue
        end
        rate_a = Float64(get(branch, "rate_a", 0.0))
        if !isfinite(rate_a) || rate_a <= 0.0
            continue
        end
        sol = solution_branches[string(branch_id)]
        pf = Float64(get(sol, "pf", 0.0))
        qf = Float64(get(sol, "qf", 0.0))
        pt = Float64(get(sol, "pt", 0.0))
        qt = Float64(get(sol, "qt", 0.0))
        s_from = sqrt(pf^2 + qf^2)
        s_to = sqrt(pt^2 + qt^2)
        loading = max(s_from, s_to) / rate_a
        push!(rows, (branch_id, Int(branch["f_bus"]), Int(branch["t_bus"]), rate_a, pf, qf, pt, qt, loading, status_value))
    end
    for (bus_id_text, bus) in sort(collect(result["solution"]["bus"]); by = x -> parse(Int, x[1]))
        bus_id = parse(Int, bus_id_text)
        push!(bus_rows, (bus_id, Float64(get(bus, "vm", 0.0)), Float64(get(bus, "va", 0.0))))
    end
    for (gen_id_text, gen) in sort(collect(result["solution"]["gen"]); by = x -> parse(Int, x[1]))
        gen_id = parse(Int, gen_id_text)
        push!(gen_rows, (gen_id, Float64(get(gen, "pg", 0.0)), Float64(get(gen, "qg", 0.0))))
    end
end

open(output_branch_csv, "w") do io
    println(io, "branch_id,f_bus,t_bus,rate_a,pf,qf,pt,qt,ac_loading,br_status")
    for row in rows
        println(io, join(row, ","))
    end
end

bus_csv = joinpath(dirname(output_branch_csv), "reference_a_bus_state.csv")
gen_csv = joinpath(dirname(output_branch_csv), "reference_a_gen_dispatch.csv")

open(bus_csv, "w") do io
    println(io, "bus_id,vm,va")
    for row in bus_rows
        println(io, join(row, ","))
    end
end

open(gen_csv, "w") do io
    println(io, "gen_id,pg,qg")
    for row in gen_rows
        println(io, join(row, ","))
    end
end

summary = Dict(
    "ac_reference_type" => "fixed_z_alpha_economic_ac_opf_reference_a",
    "case_path" => case_path,
    "alpha_effective_csv" => alpha_effective_csv,
    "offline_branch_ids" => offline_branch_ids_arg,
    "termination_status" => status,
    "objective" => objective,
    "runtime_seconds" => runtime,
    "baseMVA" => data["baseMVA"],
    "load_count" => length(data["load"]),
    "missing_alpha_load_count" => length(missing_load_ids),
    "offline_branch_count_requested" => length(offline_branch_ids),
    "exported_active_branch_count" => length(rows),
    "exported_bus_count" => length(bus_rows),
    "exported_gen_count" => length(gen_rows),
    "bus_state_csv" => bus_csv,
    "gen_dispatch_csv" => gen_csv,
    "max_ac_loading" => isempty(rows) ? nothing : maximum(row[9] for row in rows),
    "num_ac_loading_gt_1" => sum(row[9] > 1.0 for row in rows),
    "loading_definition" => "max(sqrt(pf^2+qf^2),sqrt(pt^2+qt^2))/rate_a",
)

json_keys = [
    "ac_reference_type",
    "case_path",
    "alpha_effective_csv",
    "offline_branch_ids",
    "termination_status",
    "objective",
    "runtime_seconds",
    "baseMVA",
    "load_count",
    "missing_alpha_load_count",
    "offline_branch_count_requested",
    "exported_active_branch_count",
    "exported_bus_count",
    "exported_gen_count",
    "bus_state_csv",
    "gen_dispatch_csv",
    "max_ac_loading",
    "num_ac_loading_gt_1",
    "loading_definition",
]

open(output_summary_json, "w") do io
    println(io, "{")
    for (idx, key) in enumerate(json_keys)
        comma = idx == length(json_keys) ? "" : ","
        println(io, "  \"", key, "\": ", json_value(summary[key]), comma)
    end
    println(io, "}")
end

println("reference_a_status ", status)
println("objective ", objective)
println("runtime_seconds ", runtime)
