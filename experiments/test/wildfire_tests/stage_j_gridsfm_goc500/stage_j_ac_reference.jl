using Ipopt
using JuMP
using PowerModels

if length(ARGS) < 7
    println("Usage: julia stage_j_ac_reference.jl <mode> <case_path> <alpha_csv> <offline_branch_ids> <source_less_load_ids> <output_dir> <start_dir_or_empty> [load_shed_limit]")
    exit(2)
end

mode = ARGS[1]
case_path = ARGS[2]
alpha_csv = ARGS[3]
offline_branch_ids_arg = ARGS[4]
source_less_load_ids_arg = ARGS[5]
output_dir = ARGS[6]
start_dir = ARGS[7]
load_shed_limit = length(ARGS) >= 8 && strip(ARGS[8]) != "" ? parse(Float64, ARGS[8]) : nothing

PowerModels.silence()
mkpath(output_dir)

function parse_id_set(value)
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

function parse_state_csv(path)
    rows = Dict{String, Dict{String, Float64}}()
    if path == "" || !isfile(path)
        return rows
    end
    lines = readlines(path)
    if isempty(lines)
        return rows
    end
    header = split(strip(lines[1]), ",")
    for line in lines[2:end]
        parts = split(strip(line), ",")
        if length(parts) < 1 || strip(parts[1]) == ""
            continue
        end
        row = Dict{String, Float64}()
        for idx in 2:min(length(header), length(parts))
            if strip(parts[idx]) != ""
                row[strip(header[idx])] = parse(Float64, strip(parts[idx]))
            end
        end
        rows[string(parse(Int, strip(parts[1])))] = row
    end
    return rows
end

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa AbstractString
        escaped = replace(
            value,
            "\\" => "\\\\",
            "\"" => "\\\"",
            "\n" => "\\n",
            "\r" => "\\r",
            "\t" => "\\t",
        )
        return "\"" * escaped * "\""
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Number
        return isfinite(value) ? string(value) : "null"
    else
        return "\"" * replace(string(value), "\\" => "\\\\", "\"" => "\\\"") * "\""
    end
end

function write_json(path, payload, keys)
    open(path, "w") do io
        println(io, "{")
        for (idx, key) in enumerate(keys)
            comma = idx == length(keys) ? "" : ","
            println(io, "  \"", key, "\": ", json_value(payload[key]), comma)
        end
        println(io, "}")
    end
end

function apply_topology!(data, offline_branch_ids)
    for branch_id in offline_branch_ids
        if haskey(data["branch"], branch_id)
            data["branch"][branch_id]["br_status"] = 0
        end
    end
end

function apply_fixed_alpha!(data, alpha)
    missing_load_ids = String[]
    for (load_id, load) in data["load"]
        if haskey(alpha, string(load_id))
            load["pd"] = Float64(load["pd"]) * alpha[string(load_id)]
            load["qd"] = Float64(load["qd"]) * alpha[string(load_id)]
        else
            push!(missing_load_ids, string(load_id))
        end
    end
    return missing_load_ids
end

function apply_reference_b_generator_bounds!(data)
    for (_, gen) in data["gen"]
        gen["pmin"] = 0.0
    end
end

function apply_start_values!(data, start_dir)
    if strip(start_dir) == "" || !isdir(start_dir)
        return "none"
    end
    bus_rows = parse_state_csv(joinpath(start_dir, "bus_start.csv"))
    gen_rows = parse_state_csv(joinpath(start_dir, "gen_start.csv"))
    for (bus_id, row) in bus_rows
        if haskey(data["bus"], bus_id)
            if haskey(row, "vm")
                data["bus"][bus_id]["vm_start"] = row["vm"]
            end
            if haskey(row, "va")
                data["bus"][bus_id]["va_start"] = row["va"]
            end
        end
    end
    for (gen_id, row) in gen_rows
        if haskey(data["gen"], gen_id)
            if haskey(row, "pg")
                data["gen"][gen_id]["pg_start"] = row["pg"]
            end
            if haskey(row, "qg")
                data["gen"][gen_id]["qg_start"] = row["qg"]
            end
        end
    end
    return "csv_start_values"
end

function export_solution(result, data, output_dir, prefix)
    has_solution = haskey(result, "solution") && haskey(result["solution"], "branch")
    branch_csv = joinpath(output_dir, "$(prefix)_branch_state.csv")
    bus_csv = joinpath(output_dir, "$(prefix)_bus_state.csv")
    gen_csv = joinpath(output_dir, "$(prefix)_gen_dispatch.csv")
    load_csv = joinpath(output_dir, "$(prefix)_load_service.csv")

    open(branch_csv, "w") do io
        println(io, "branch_id,f_bus,t_bus,rate_a,pf,qf,pt,qt,ac_loading,br_status")
        if has_solution
            solution_branches = result["solution"]["branch"]
            for (branch_id_text, branch) in sort(collect(data["branch"]); by = x -> parse(Int, x[1]))
                branch_id = parse(Int, branch_id_text)
                status_value = Int(get(branch, "br_status", 1))
                if status_value != 1 || !haskey(solution_branches, string(branch_id))
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
                loading = max(sqrt(pf^2 + qf^2), sqrt(pt^2 + qt^2)) / rate_a
                println(io, join((branch_id, Int(branch["f_bus"]), Int(branch["t_bus"]), rate_a, pf, qf, pt, qt, loading, status_value), ","))
            end
        end
    end

    open(bus_csv, "w") do io
        println(io, "bus_id,vm,va")
        if has_solution && haskey(result["solution"], "bus")
            for (bus_id_text, bus) in sort(collect(result["solution"]["bus"]); by = x -> parse(Int, x[1]))
                println(io, join((parse(Int, bus_id_text), Float64(get(bus, "vm", 0.0)), Float64(get(bus, "va", 0.0))), ","))
            end
        end
    end

    open(gen_csv, "w") do io
        println(io, "gen_id,pg,qg")
        if has_solution && haskey(result["solution"], "gen")
            for (gen_id_text, gen) in sort(collect(result["solution"]["gen"]); by = x -> parse(Int, x[1]))
                println(io, join((parse(Int, gen_id_text), Float64(get(gen, "pg", 0.0)), Float64(get(gen, "qg", 0.0))), ","))
            end
        end
    end

    open(load_csv, "w") do io
        println(io, "load_id,status,pd,qd")
        if has_solution && haskey(result["solution"], "load")
            for (load_id_text, load) in sort(collect(result["solution"]["load"]); by = x -> parse(Int, x[1]))
                println(io, join((parse(Int, load_id_text), Float64(get(load, "status", 1.0)), Float64(get(load, "pd", 0.0)), Float64(get(load, "qd", 0.0))), ","))
            end
        end
    end

    return branch_csv, bus_csv, gen_csv, load_csv
end

function constraint_power_balance_load_only(pm, i; nw=PowerModels.nw_id_default)
    bus_arcs = PowerModels.ref(pm, nw, :bus_arcs, i)
    bus_arcs_dc = PowerModels.ref(pm, nw, :bus_arcs_dc, i)
    bus_arcs_sw = PowerModels.ref(pm, nw, :bus_arcs_sw, i)
    bus_gens = PowerModels.ref(pm, nw, :bus_gens, i)
    bus_loads = PowerModels.ref(pm, nw, :bus_loads, i)
    bus_shunts = PowerModels.ref(pm, nw, :bus_shunts, i)
    bus_storage = PowerModels.ref(pm, nw, :bus_storage, i)
    bus_pd = Dict(k => PowerModels.ref(pm, nw, :load, k, "pd") for k in bus_loads)
    bus_qd = Dict(k => PowerModels.ref(pm, nw, :load, k, "qd") for k in bus_loads)
    bus_gs = Dict(k => PowerModels.ref(pm, nw, :shunt, k, "gs") for k in bus_shunts)
    bus_bs = Dict(k => PowerModels.ref(pm, nw, :shunt, k, "bs") for k in bus_shunts)

    vm = PowerModels.var(pm, nw, :vm, i)
    p = get(PowerModels.var(pm, nw), :p, Dict())
    q = get(PowerModels.var(pm, nw), :q, Dict())
    pg = get(PowerModels.var(pm, nw), :pg, Dict())
    qg = get(PowerModels.var(pm, nw), :qg, Dict())
    ps = get(PowerModels.var(pm, nw), :ps, Dict())
    qs = get(PowerModels.var(pm, nw), :qs, Dict())
    psw = get(PowerModels.var(pm, nw), :psw, Dict())
    qsw = get(PowerModels.var(pm, nw), :qsw, Dict())
    p_dc = get(PowerModels.var(pm, nw), :p_dc, Dict())
    q_dc = get(PowerModels.var(pm, nw), :q_dc, Dict())
    z_demand = PowerModels.var(pm, nw, :z_demand)

    JuMP.@constraint(pm.model,
        sum(p[a] for a in bus_arcs)
        + sum(p_dc[a_dc] for a_dc in bus_arcs_dc)
        + sum(psw[a_sw] for a_sw in bus_arcs_sw)
        ==
        sum(pg[g] for g in bus_gens)
        - sum(ps[s] for s in bus_storage)
        - sum(pd*z_demand[k] for (k,pd) in bus_pd)
        - sum(gs for (_,gs) in bus_gs)*vm^2
    )
    JuMP.@constraint(pm.model,
        sum(q[a] for a in bus_arcs)
        + sum(q_dc[a_dc] for a_dc in bus_arcs_dc)
        + sum(qsw[a_sw] for a_sw in bus_arcs_sw)
        ==
        sum(qg[g] for g in bus_gens)
        - sum(qs[s] for s in bus_storage)
        - sum(qd*z_demand[k] for (k,qd) in bus_qd)
        + sum(bs for (_,bs) in bus_bs)*vm^2
    )
end

function build_ref_b_mld(pm)
    PowerModels.variable_bus_voltage(pm)
    PowerModels.variable_gen_power(pm)
    PowerModels.variable_branch_power(pm)
    PowerModels.variable_dcline_power(pm)
    PowerModels.variable_load_power_factor(pm, relax=true)

    z_demand = PowerModels.var(pm, :z_demand)
    for load_id in PowerModels.ids(pm, :load)
        if string(load_id) in SOURCE_LESS_LOAD_IDS
            JuMP.@constraint(pm.model, z_demand[load_id] == 0.0)
        end
    end
    JuMP.@objective(pm.model, Max,
        sum(abs(load["pd"]) * z_demand[i] for (i, load) in PowerModels.ref(pm, :load))
    )

    PowerModels.constraint_model_voltage(pm)
    for i in PowerModels.ids(pm, :ref_buses)
        PowerModels.constraint_theta_ref(pm, i)
    end
    for i in PowerModels.ids(pm, :bus)
        constraint_power_balance_load_only(pm, i)
    end
    for i in PowerModels.ids(pm, :branch)
        PowerModels.constraint_ohms_yt_from(pm, i)
        PowerModels.constraint_ohms_yt_to(pm, i)
        PowerModels.constraint_voltage_angle_difference(pm, i)
        PowerModels.constraint_thermal_limit_from(pm, i)
        PowerModels.constraint_thermal_limit_to(pm, i)
    end
    for i in PowerModels.ids(pm, :dcline)
        PowerModels.constraint_dcline_power_losses(pm, i)
    end
end

function build_ref_b_cost_tiebreak(pm)
    build_ref_b_mld(pm)
    z_demand = PowerModels.var(pm, :z_demand)
    served = sum(abs(load["pd"]) * z_demand[i] for (i, load) in PowerModels.ref(pm, :load))
    total = sum(abs(load["pd"]) for (_, load) in PowerModels.ref(pm, :load))
    if LOAD_SHED_LIMIT !== nothing
        JuMP.@constraint(pm.model, total - served <= LOAD_SHED_LIMIT)
    end
    PowerModels.objective_min_fuel_cost(pm)
end

function solve_reference_a()
    data = PowerModels.parse_file(case_path; validate=true)
    alpha = parse_alpha_csv(alpha_csv)
    offline_branch_ids = parse_id_set(offline_branch_ids_arg)
    missing_load_ids = apply_fixed_alpha!(data, alpha)
    apply_topology!(data, offline_branch_ids)
    start_source = apply_start_values!(data, start_dir)
    solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 10000, "mu_init" => 1.0, "warm_start_bound_push" => 1.0)
    t0 = time()
    result = PowerModels.solve_ac_opf(data, solver)
    runtime = time() - t0
    branch_csv, bus_csv, gen_csv, load_csv = export_solution(result, data, output_dir, "reference_a")
    summary = Dict(
        "mode" => "reference_a_fixed_z_alpha_economic_ac_opf",
        "termination_status" => string(result["termination_status"]),
        "objective" => haskey(result, "objective") ? result["objective"] : nothing,
        "runtime_seconds" => runtime,
        "iteration_count" => nothing,
        "start_source" => start_source,
        "missing_alpha_load_count" => length(missing_load_ids),
        "offline_branch_ids" => offline_branch_ids_arg,
        "branch_state_csv" => branch_csv,
        "bus_state_csv" => bus_csv,
        "gen_dispatch_csv" => gen_csv,
        "load_service_csv" => load_csv,
    )
    write_json(joinpath(output_dir, "reference_a_summary.json"), summary, collect(keys(summary)))
    println("reference_a_status ", summary["termination_status"])
end

function solve_reference_b(builder, prefix)
    data = PowerModels.parse_file(case_path; validate=true)
    offline_branch_ids = parse_id_set(offline_branch_ids_arg)
    apply_topology!(data, offline_branch_ids)
    apply_reference_b_generator_bounds!(data)
    solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 10000)
    t0 = time()
    result = PowerModels.solve_model(data, ACPPowerModel, solver, builder)
    runtime = time() - t0
    branch_csv, bus_csv, gen_csv, load_csv = export_solution(result, data, output_dir, prefix)
    summary = Dict(
        "mode" => prefix,
        "termination_status" => string(result["termination_status"]),
        "objective" => haskey(result, "objective") ? result["objective"] : nothing,
        "runtime_seconds" => runtime,
        "iteration_count" => nothing,
        "reference_b_pg_min_convention" => "0<=Pg<=Pgmax",
        "reference_b_shunt_policy" => "fixed_shunts_no_z_shunt_variable",
        "offline_branch_ids" => offline_branch_ids_arg,
        "branch_state_csv" => branch_csv,
        "bus_state_csv" => bus_csv,
        "gen_dispatch_csv" => gen_csv,
        "load_service_csv" => load_csv,
    )
    write_json(joinpath(output_dir, "$(prefix)_summary.json"), summary, collect(keys(summary)))
    println(prefix, "_status ", summary["termination_status"])
end

const SOURCE_LESS_LOAD_IDS = parse_id_set(source_less_load_ids_arg)
const LOAD_SHED_LIMIT = load_shed_limit

try
    if mode == "reference_a"
        solve_reference_a()
    elseif mode == "reference_b_b1"
        solve_reference_b(build_ref_b_mld, "reference_b_b1")
    elseif mode == "reference_b_b2"
        solve_reference_b(build_ref_b_cost_tiebreak, "reference_b_b2")
    else
        error("unknown mode: $(mode)")
    end
catch err
    summary = Dict(
        "mode" => mode,
        "termination_status" => "STAGE_J_SCRIPT_EXCEPTION",
        "objective" => nothing,
        "runtime_seconds" => nothing,
        "iteration_count" => nothing,
        "message" => sprint(showerror, err),
    )
    write_json(joinpath(output_dir, "$(mode)_summary.json"), summary, collect(keys(summary)))
    rethrow(err)
end
