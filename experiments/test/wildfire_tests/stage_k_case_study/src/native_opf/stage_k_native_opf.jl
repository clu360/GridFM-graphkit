using Ipopt
using JuMP
using PowerModels

if length(ARGS) < 9
    println("Usage: julia stage_k_native_opf.jl <mode> <case.m> <offline_pm_ids> <source_less_load_ids> <p_env_csv> <alpha_csv_or_dash> <lambda_r> <r_base> <output_dir> [service_tolerance_mw]")
    exit(2)
end

const MODE = ARGS[1]
const CASE_PATH = ARGS[2]
const OFFLINE_IDS_ARG = ARGS[3]
const SOURCELESS_IDS_ARG = ARGS[4]
const P_ENV_CSV = ARGS[5]
const ALPHA_CSV = ARGS[6]
const LAMBDA_R = parse(Float64, ARGS[7])
const R_BASE = parse(Float64, ARGS[8])
const OUTPUT_DIR = ARGS[9]
const SERVICE_TOLERANCE_MW = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1e-4

PowerModels.silence()
mkpath(OUTPUT_DIR)

parse_ids(value) = Set(parse(Int, strip(x)) for x in split(replace(value, "," => ";"), ";") if !isempty(strip(x)))

function parse_two_column_csv(path)
    values = Dict{Int,Float64}()
    for (idx, line) in enumerate(readlines(path))
        parts = split(strip(line), ",")
        if idx == 1 || length(parts) < 2 || isempty(strip(parts[1]))
            continue
        end
        values[parse(Int, strip(parts[1]))] = parse(Float64, strip(parts[2]))
    end
    return values
end

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa AbstractString
        return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Number
        return isfinite(value) ? string(value) : "null"
    else
        return json_value(string(value))
    end
end

function write_json(path, payload)
    keys_sorted = sort(collect(keys(payload)))
    open(path, "w") do io
        println(io, "{")
        for (idx, key) in enumerate(keys_sorted)
            comma = idx == length(keys_sorted) ? "" : ","
            println(io, "  ", json_value(key), ": ", json_value(payload[key]), comma)
        end
        println(io, "}")
    end
end

function apply_topology!(data, offline_ids)
    missing = Int[]
    for branch_id in offline_ids
        key = string(branch_id)
        if haskey(data["branch"], key)
            data["branch"][key]["br_status"] = 0
        else
            push!(missing, branch_id)
        end
    end
    isempty(missing) || error("offline PowerModels branch IDs missing: $(missing)")
end

function apply_reference_b_generator_bounds!(data)
    for (_, gen) in data["gen"]
        if Int(get(gen, "gen_status", 1)) == 1
            gen["pmin"] = 0.0
            gen["pmax"] = max(0.0, Float64(gen["pmax"]))
        end
    end
end

function clamp_sourceless!(pm, source_less)
    z = PowerModels.var(pm, :z_demand)
    for load_id in PowerModels.ids(pm, :load)
        if load_id in source_less
            JuMP.@constraint(pm.model, z[load_id] == 0.0)
        end
    end
end

function active_service_terms(pm)
    z = PowerModels.var(pm, :z_demand)
    total = sum(abs(Float64(load["pd"])) for (_, load) in PowerModels.ref(pm, :load))
    served = sum(abs(Float64(load["pd"])) * z[i] for (i, load) in PowerModels.ref(pm, :load))
    return z, total, served
end

function constraint_power_balance_ac_service(pm, i; nw=PowerModels.nw_id_default)
    bus_arcs = PowerModels.ref(pm, nw, :bus_arcs, i)
    bus_arcs_dc = PowerModels.ref(pm, nw, :bus_arcs_dc, i)
    bus_arcs_sw = PowerModels.ref(pm, nw, :bus_arcs_sw, i)
    bus_gens = PowerModels.ref(pm, nw, :bus_gens, i)
    bus_loads = PowerModels.ref(pm, nw, :bus_loads, i)
    bus_shunts = PowerModels.ref(pm, nw, :bus_shunts, i)
    bus_storage = PowerModels.ref(pm, nw, :bus_storage, i)
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
    vm = PowerModels.var(pm, nw, :vm, i)
    z = PowerModels.var(pm, nw, :z_demand)
    JuMP.@constraint(pm.model,
        sum(p[a] for a in bus_arcs) + sum(p_dc[a] for a in bus_arcs_dc) + sum(psw[a] for a in bus_arcs_sw)
        == sum(pg[g] for g in bus_gens) - sum(ps[s] for s in bus_storage)
        - sum(PowerModels.ref(pm, nw, :load, k, "pd") * z[k] for k in bus_loads)
        - sum(PowerModels.ref(pm, nw, :shunt, s, "gs") for s in bus_shunts) * vm^2)
    JuMP.@constraint(pm.model,
        sum(q[a] for a in bus_arcs) + sum(q_dc[a] for a in bus_arcs_dc) + sum(qsw[a] for a in bus_arcs_sw)
        == sum(qg[g] for g in bus_gens) - sum(qs[s] for s in bus_storage)
        - sum(PowerModels.ref(pm, nw, :load, k, "qd") * z[k] for k in bus_loads)
        + sum(PowerModels.ref(pm, nw, :shunt, s, "bs") for s in bus_shunts) * vm^2)
end

function constraint_power_balance_dc_service(pm, i; nw=PowerModels.nw_id_default)
    bus_arcs = PowerModels.ref(pm, nw, :bus_arcs, i)
    bus_gens = PowerModels.ref(pm, nw, :bus_gens, i)
    bus_loads = PowerModels.ref(pm, nw, :bus_loads, i)
    p = PowerModels.var(pm, nw, :p)
    pg = PowerModels.var(pm, nw, :pg)
    z = PowerModels.var(pm, nw, :z_demand)
    JuMP.@constraint(pm.model,
        sum(p[a] for a in bus_arcs) == sum(pg[g] for g in bus_gens)
        - sum(PowerModels.ref(pm, nw, :load, k, "pd") * z[k] for k in bus_loads))
end

function add_network_constraints(pm; ac)
    PowerModels.constraint_model_voltage(pm)
    for i in PowerModels.ids(pm, :ref_buses)
        PowerModels.constraint_theta_ref(pm, i)
    end
    for i in PowerModels.ids(pm, :bus)
        ac ? constraint_power_balance_ac_service(pm, i) : constraint_power_balance_dc_service(pm, i)
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

function build_native(pm; ac)
    PowerModels.variable_bus_voltage(pm)
    PowerModels.variable_gen_power(pm)
    PowerModels.variable_branch_power(pm)
    PowerModels.variable_dcline_power(pm)
    PowerModels.variable_load_power_factor(pm, relax=true)
    clamp_sourceless!(pm, SOURCELESS_IDS)
    add_network_constraints(pm; ac=ac)
    _, total, served = active_service_terms(pm)
    p = PowerModels.var(pm, :p)
    q = ac ? PowerModels.var(pm, :q) : nothing
    risk_terms = Any[]
    epigraph = Dict{Int,VariableRef}()
    for (branch_id, probability) in P_ENV
        haskey(PowerModels.ref(pm, :branch), branch_id) || continue
        branch = PowerModels.ref(pm, :branch, branch_id)
        Int(get(branch, "br_status", 1)) == 1 || continue
        rate = Float64(branch["rate_a"])
        rate > 0.0 || error("risk branch $branch_id has nonpositive rate_a")
        f = Int(branch["f_bus"]); t = Int(branch["t_bus"])
        if ac
            epigraph[branch_id] = JuMP.@variable(pm.model, lower_bound=0.0, base_name="risk_t[$branch_id]")
            JuMP.@constraint(pm.model, epigraph[branch_id] >= (p[(branch_id,f,t)]^2 + q[(branch_id,f,t)]^2) / rate^2)
            JuMP.@constraint(pm.model, epigraph[branch_id] >= (p[(branch_id,t,f)]^2 + q[(branch_id,t,f)]^2) / rate^2)
            push!(risk_terms, probability * epigraph[branch_id])
        else
            push!(risk_terms, probability * p[(branch_id,f,t)]^2 / rate^2)
        end
    end
    risk_norm = sum(risk_terms) / R_BASE
    load_shed = (total - served) / total
    JuMP.@objective(pm.model, Min, LAMBDA_R * risk_norm + (1.0 - LAMBDA_R) * load_shed)
    pm.ext[:stage_k_total_pd] = total
    pm.ext[:stage_k_risk_epigraph] = epigraph
end

build_native_ac(pm) = build_native(pm; ac=true)
build_native_dc(pm) = build_native(pm; ac=false)

function build_reference_b(pm; economic=false)
    PowerModels.variable_bus_voltage(pm)
    PowerModels.variable_gen_power(pm)
    PowerModels.variable_branch_power(pm)
    PowerModels.variable_dcline_power(pm)
    PowerModels.variable_load_power_factor(pm, relax=true)
    clamp_sourceless!(pm, SOURCELESS_IDS)
    add_network_constraints(pm; ac=true)
    _, total, served = active_service_terms(pm)
    if economic
        JuMP.@constraint(pm.model, total - served <= LOAD_SHED_LIMIT[])
        PowerModels.objective_min_fuel_cost(pm)
    else
        JuMP.@objective(pm.model, Max, served)
    end
end

build_reference_b1(pm) = build_reference_b(pm; economic=false)
build_reference_b2(pm) = build_reference_b(pm; economic=true)

function export_result(pm, result, data, prefix)
    branch_path = joinpath(OUTPUT_DIR, prefix * "_branch_state.csv")
    load_path = joinpath(OUTPUT_DIR, prefix * "_load_service.csv")
    bus_path = joinpath(OUTPUT_DIR, prefix * "_bus_state.csv")
    gen_path = joinpath(OUTPUT_DIR, prefix * "_gen_state.csv")
    has_solution = haskey(result, "solution")
    open(branch_path, "w") do io
        println(io, "powermodels_branch_id,pf,qf,pt,qt,physical_loading,risk_epigraph")
        has_solution || return
        for (id_text, branch) in sort(collect(data["branch"]); by=x->parse(Int,x[1]))
            id = parse(Int, id_text)
            Int(get(branch, "br_status", 1)) == 1 || continue
            haskey(result["solution"]["branch"], id_text) || continue
            sol = result["solution"]["branch"][id_text]
            pf = Float64(get(sol,"pf",0.0)); qf = Float64(get(sol,"qf",0.0))
            pt = Float64(get(sol,"pt",0.0)); qt = Float64(get(sol,"qt",0.0))
            rate = Float64(get(branch,"rate_a",0.0))
            apparent = MODE == "native_dc" ? max(abs(pf), abs(pt)) : max(sqrt(pf^2+qf^2), sqrt(pt^2+qt^2))
            loading = rate > 0 ? apparent/rate : NaN
            epi = ""
            if haskey(pm.ext, :stage_k_risk_epigraph) && haskey(pm.ext[:stage_k_risk_epigraph], id)
                epi = string(JuMP.value(pm.ext[:stage_k_risk_epigraph][id]))
            end
            println(io, join((id,pf,qf,pt,qt,loading,epi),","))
        end
    end
    open(load_path, "w") do io
        println(io, "load_id,service_fraction,pd_requested,qd_requested")
        has_solution || return
        solution_loads = get(result["solution"], "load", Dict())
        for (id_text, load) in sort(collect(data["load"]); by=x->parse(Int,x[1]))
            service = haskey(solution_loads,id_text) ? Float64(get(solution_loads[id_text],"status",1.0)) : 1.0
            println(io, join((parse(Int,id_text),service,load["pd"],load["qd"]),","))
        end
    end
    open(bus_path, "w") do io
        println(io, "bus_id,vm,va")
        has_solution || return
        for (id_text, bus) in sort(collect(get(result["solution"],"bus",Dict())); by=x->parse(Int,x[1]))
            println(io, join((parse(Int,id_text),get(bus,"vm",""),get(bus,"va","")),","))
        end
    end
    open(gen_path, "w") do io
        println(io, "gen_id,pg,qg")
        has_solution || return
        for (id_text, gen) in sort(collect(get(result["solution"],"gen",Dict())); by=x->parse(Int,x[1]))
            println(io, join((parse(Int,id_text),get(gen,"pg",""),get(gen,"qg","")),","))
        end
    end
    return branch_path, load_path, bus_path, gen_path
end

function diagnostic_generation_cost(result, data)
    haskey(result, "solution") && haskey(result["solution"], "gen") || return nothing
    total = 0.0
    for (id_text, sol) in result["solution"]["gen"]
        haskey(data["gen"], id_text) || continue
        gen = data["gen"][id_text]
        Int(get(gen, "model", 2)) == 2 || return nothing
        coefficients = Float64.(get(gen, "cost", Float64[]))
        isempty(coefficients) && return nothing
        pg = Float64(get(sol, "pg", 0.0))
        degree = length(coefficients) - 1
        total += sum(coefficients[idx] * pg^(degree - idx + 1) for idx in eachindex(coefficients))
    end
    return total
end

function solve_model_mode(data, model_type, builder, prefix)
    solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "max_iter"=>10000)
    pm = PowerModels.instantiate_model(data, model_type, builder)
    started = time()
    result = PowerModels.optimize_model!(pm; optimizer=solver)
    elapsed = time() - started
    branch_path, load_path, bus_path, gen_path = export_result(pm, result, data, prefix)
    iterations = try Int(JuMP.barrier_iterations(pm.model)) catch; nothing end
    summary = Dict{String,Any}(
        "mode"=>MODE,
        "termination_status"=>string(result["termination_status"]),
        "objective"=>get(result,"objective",nothing),
        "elapsed_seconds"=>elapsed,
        "solver_time_seconds"=>get(result,"solve_time",nothing),
        "iteration_count"=>iterations,
        "branch_state_csv"=>branch_path,
        "load_service_csv"=>load_path,
        "bus_state_csv"=>bus_path,
        "gen_state_csv"=>gen_path,
        "risk_epigraph_count"=>length(get(pm.ext,:stage_k_risk_epigraph,Dict())),
        "risk_line_count_expected"=>count(id -> !(id in OFFLINE_IDS), keys(P_ENV)),
        "economic_cost_diagnostic"=>diagnostic_generation_cost(result,data),
    )
    write_json(joinpath(OUTPUT_DIR,prefix*"_summary.json"),summary)
    return result, summary
end

const OFFLINE_IDS = parse_ids(OFFLINE_IDS_ARG)
const SOURCELESS_IDS = parse_ids(SOURCELESS_IDS_ARG)
const P_ENV = parse_two_column_csv(P_ENV_CSV)
const LOAD_SHED_LIMIT = Ref{Float64}(0.0)

data = PowerModels.parse_file(CASE_PATH; validate=true)
apply_topology!(data, OFFLINE_IDS)

try
    if MODE == "baseline_audit"
        solver = optimizer_with_attributes(Ipopt.Optimizer,"print_level"=>0,"max_iter"=>10000)
        started=time(); result=PowerModels.solve_ac_opf(data,solver); elapsed=time()-started
        pm = PowerModels.instantiate_model(data,PowerModels.ACPPowerModel,PowerModels.build_opf)
        branch_path,load_path,bus_path,gen_path=export_result(pm,result,data,"baseline_audit")
        write_json(joinpath(OUTPUT_DIR,"baseline_audit_summary.json"),Dict(
            "mode"=>MODE,"termination_status"=>string(result["termination_status"]),
            "objective"=>get(result,"objective",nothing),"elapsed_seconds"=>elapsed,
            "solver_time_seconds"=>get(result,"solve_time",nothing),"branch_state_csv"=>branch_path,
            "load_service_csv"=>load_path,"bus_state_csv"=>bus_path,"gen_state_csv"=>gen_path))
    elseif MODE == "native_ac"
        solve_model_mode(data, PowerModels.ACPPowerModel, build_native_ac, "native_ac")
    elseif MODE == "native_dc"
        solve_model_mode(data, PowerModels.DCPPowerModel, build_native_dc, "native_dc")
    elseif MODE == "reference_a"
        alpha = parse_two_column_csv(ALPHA_CSV)
        for (load_id, load) in data["load"]
            id = parse(Int,load_id)
            haskey(alpha,id) || error("Reference A missing alpha for load $id")
            load["pd"] *= alpha[id]; load["qd"] *= alpha[id]
        end
        solver = optimizer_with_attributes(Ipopt.Optimizer,"print_level"=>0,"max_iter"=>10000)
        started=time(); result=PowerModels.solve_ac_opf(data,solver); elapsed=time()-started
        # Instantiate solely for a uniform export container is not valid here;
        # write the economic result through a standard ACP model container.
        pm = PowerModels.instantiate_model(data,PowerModels.ACPPowerModel,PowerModels.build_opf)
        branch_path,load_path,bus_path,gen_path=export_result(pm,result,data,"reference_a")
        write_json(joinpath(OUTPUT_DIR,"reference_a_summary.json"),Dict(
            "mode"=>MODE,"termination_status"=>string(result["termination_status"]),
            "objective"=>get(result,"objective",nothing),"elapsed_seconds"=>elapsed,
            "solver_time_seconds"=>get(result,"solve_time",nothing),"branch_state_csv"=>branch_path,
            "load_service_csv"=>load_path,"bus_state_csv"=>bus_path,"gen_state_csv"=>gen_path))
    elseif MODE == "reference_b"
        apply_reference_b_generator_bounds!(data)
        result_b1, summary_b1 = solve_model_mode(data,PowerModels.ACPPowerModel,build_reference_b1,"reference_b_b1")
        haskey(result_b1,"solution") || error("Reference B1 returned no solution")
        load_rows = get(result_b1["solution"],"load",Dict())
        total = sum(abs(Float64(load["pd"])) for (_,load) in data["load"])
        served = sum(abs(Float64(data["load"][id]["pd"])) * Float64(get(row,"status",1.0)) for (id,row) in load_rows)
        LOAD_SHED_LIMIT[] = max(0.0,total-served) + SERVICE_TOLERANCE_MW / Float64(data["baseMVA"])
        result_b2, summary_b2 = solve_model_mode(data,PowerModels.ACPPowerModel,build_reference_b2,"reference_b_b2")
        write_json(joinpath(OUTPUT_DIR,"reference_b_summary.json"),Dict(
            "mode"=>MODE,"b1_status"=>summary_b1["termination_status"],
            "b2_status"=>summary_b2["termination_status"],"maximum_served_pu"=>served,
            "load_shed_limit_pu"=>LOAD_SHED_LIMIT[]))
    else
        error("unknown Stage K mode: $MODE")
    end
catch err
    write_json(joinpath(OUTPUT_DIR,MODE*"_exception.json"),Dict(
        "mode"=>MODE,"termination_status"=>"evaluator_exception","message"=>sprint(showerror,err)))
    rethrow(err)
end
