using Pkg

Pkg.add(["JuMP", "PowerModels", "Ipopt"])
Pkg.precompile()

using JuMP
using Ipopt
using PowerModels

deps = Pkg.dependencies()

function pkg_version(uuid::String)
    dep = deps[Base.UUID(uuid)]
    return dep.version
end

println("Julia ", VERSION)
println("JuMP ", pkg_version("4076af6c-e467-56ae-b986-b466b2749572"))
println("Ipopt ", pkg_version("b6b21f68-93f8-5de0-b562-5493be1d77c9"))
println("PowerModels ", pkg_version("c36e90e8-916a-50a6-bd94-075b64ef4655"))

model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x)
@objective(model, Min, (x - 2)^2)
optimize!(model)

println("termination_status ", termination_status(model))
println("objective ", objective_value(model))
println("x ", value(x))
