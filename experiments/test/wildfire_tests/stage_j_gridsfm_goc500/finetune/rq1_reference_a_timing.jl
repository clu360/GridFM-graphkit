using Ipopt
using JuMP
using PowerModels

if length(ARGS) != 6
    println("Usage: julia rq1_reference_a_timing.jl <case> <manifest_tsv> <output_csv> <warmups> <repetitions> <status_json>")
    exit(2)
end

case_path = ARGS[1]
manifest_path = ARGS[2]
output_path = ARGS[3]
warmup_count = parse(Int, ARGS[4])
repetitions = parse(Int, ARGS[5])
status_path = ARGS[6]

PowerModels.silence()

function parse_table(path, delimiter='\t')
    lines = readlines(path)
    isempty(lines) && error("empty manifest: $(path)")
    header = split(lines[1], delimiter)
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, delimiter; keepempty=true)
        length(values) == length(header) || error("manifest width mismatch")
        push!(rows, Dict(strip(header[i]) => strip(values[i]) for i in eachindex(header)))
    end
    return rows
end

function parse_alpha_csv(path)
    alpha = Dict{String, Float64}()
    for (index, line) in enumerate(readlines(path))
        index == 1 && continue
        parts = split(strip(line), ",")
        length(parts) >= 2 || continue
        alpha[string(parse(Int, strip(parts[1])))] = parse(Float64, strip(parts[2]))
    end
    return alpha
end

function parse_id_set(value)
    ids = Set{String}()
    for part in split(replace(value, "," => ";"), ";")
        stripped = strip(part)
        isempty(stripped) || push!(ids, string(parse(Int, stripped)))
    end
    return ids
end

function apply_topology!(data, offline_branch_ids)
    for branch_id in offline_branch_ids
        haskey(data["branch"], branch_id) && (data["branch"][branch_id]["br_status"] = 0)
    end
end

function apply_fixed_alpha!(data, alpha)
    missing = String[]
    for (load_id, load) in data["load"]
        if haskey(alpha, string(load_id))
            load["pd"] = Float64(load["pd"]) * alpha[string(load_id)]
            load["qd"] = Float64(load["qd"]) * alpha[string(load_id)]
        else
            push!(missing, string(load_id))
        end
    end
    return missing
end

function apply_standard_reference_a_start!(data)
    for (_, bus) in data["bus"]
        bus["vm_start"] = 1.0
        bus["va_start"] = 0.0
    end
    for (_, gen) in data["gen"]
        gen["pg_start"] = (Float64(gen["pmin"]) + Float64(gen["pmax"])) / 2.0
        gen["qg_start"] = 0.0
    end
end

function finite_solution_state(result)
    solution = result["solution"]
    required = Dict(
        "bus" => ["vm", "va"],
        "gen" => ["pg", "qg"],
        "branch" => ["pf", "qf", "pt", "qt"],
    )
    count = 0
    finite = true
    for (family, fields) in required
        haskey(solution, family) || return false, count
        for (_, record) in solution[family]
            for field in fields
                if haskey(record, field)
                    value = Float64(record[field])
                    finite &= isfinite(value)
                    count += 1
                end
            end
        end
    end
    return finite, count
end

function run_one(base_data, row, repetition, position)
    total_start = time_ns()
    data = deepcopy(base_data)
    alpha = parse_alpha_csv(row["alpha_csv"])
    missing = apply_fixed_alpha!(data, alpha)
    apply_topology!(data, parse_id_set(row["offline_branch_ids"]))
    apply_standard_reference_a_start!(data)
    prepare_end = time_ns()

    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => 10000,
        "mu_init" => 1.0,
        "warm_start_bound_push" => 1.0,
    )
    pm = PowerModels.instantiate_model(data, PowerModels.ACPPowerModel, PowerModels.build_opf)
    build_end = time_ns()
    result = PowerModels.optimize_model!(pm; optimizer=solver)
    solve_end = time_ns()
    state_finite, state_scalar_count = finite_solution_state(result)
    postprocess_end = time_ns()

    iteration_count = try
        Int(JuMP.barrier_iterations(pm.model))
    catch
        -1
    end
    ipopt_time = haskey(result, "solve_time") ? Float64(result["solve_time"]) : NaN
    objective = haskey(result, "objective") ? Float64(result["objective"]) : NaN
    return Dict{String, Any}(
        "unique_id" => row["unique_id"],
        "decision_sha256" => row["decision_sha256"],
        "repetition" => repetition,
        "execution_position" => position,
        "status" => string(result["termination_status"]),
        "objective" => objective,
        "iteration_count" => iteration_count,
        "prepare_seconds" => (prepare_end - total_start) / 1e9,
        "build_seconds" => (build_end - prepare_end) / 1e9,
        "optimize_call_seconds" => (solve_end - build_end) / 1e9,
        "ipopt_solve_time_seconds" => ipopt_time,
        "postprocess_seconds" => (postprocess_end - solve_end) / 1e9,
        "evaluator_total_seconds" => (postprocess_end - total_start) / 1e9,
        "missing_alpha_load_count" => length(missing),
        "state_finite" => state_finite,
        "state_scalar_count" => state_scalar_count,
        "start_source" => "explicit_generic_V1_theta0_Pg_midpoint_Qg0",
        "timing_boundary" => "fixed_candidate_received_to_usable_electrical_state_returned",
        "downstream_scoring_included" => false,
        "publication_io_included" => false,
    )
end

function csv_value(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_rows(path, rows, fields)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(fields, ","))
        for row in rows
            println(io, join([csv_value(row[field]) for field in fields], ","))
        end
    end
end

function write_status(path, status, rows, elapsed)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"status\": \"", status, "\",")
        println(io, "  \"rows\": ", rows, ",")
        println(io, "  \"elapsed_seconds\": ", elapsed, ",")
        println(io, "  \"julia_version\": \"", VERSION, "\",")
        println(io, "  \"threads\": ", Threads.nthreads(), ",")
        println(io, "  \"persistent_process\": true,")
        println(io, "  \"base_case_parsed_once\": true")
        println(io, "}")
    end
end

manifest = parse_table(manifest_path)
isempty(manifest) && error("RQ1 manifest contains no instances")
base_data = PowerModels.parse_file(case_path; validate=true)

for index in 1:warmup_count
    run_one(base_data, manifest[mod1(index, length(manifest))], 0, index)
end

started = time()
rows = Dict{String, Any}[]
for repetition in 1:repetitions
    shift = mod(repetition - 1, length(manifest))
    ordered = vcat(manifest[(shift + 1):end], manifest[1:shift])
    for (position, row) in enumerate(ordered)
        push!(rows, run_one(base_data, row, repetition, position))
    end
end

fields = [
    "unique_id", "decision_sha256", "repetition", "execution_position", "status",
    "objective", "iteration_count", "prepare_seconds", "build_seconds",
    "optimize_call_seconds", "ipopt_solve_time_seconds", "postprocess_seconds",
    "evaluator_total_seconds", "missing_alpha_load_count", "state_finite",
    "state_scalar_count", "start_source", "timing_boundary",
    "downstream_scoring_included", "publication_io_included",
]
write_rows(output_path, rows, fields)
write_status(status_path, "RQ1_REFERENCE_A_BATCH_COMPLETE", length(rows), time() - started)
println("RQ1_REFERENCE_A_BATCH_COMPLETE rows=", length(rows))

