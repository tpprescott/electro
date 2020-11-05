
function observation_filter(sol::DESolution)
    y = zeros(ComplexF64, 37)
    observation_filter(sol, y)
end
function observation_filter(sol::DESolution, y::AbstractVector{ComplexF64})
    broadcast!(j -> sol(5*j)[2], y, 0:1:36)
    return y
end

function observation_filter(sol::EnsembleSolution)
    y = zeros(ComplexF64, 37, length(sol))
    observation_filter(sol, y)
end
function observation_filter(sol::EnsembleSolution, y::AbstractMatrix{ComplexF64})
    map(observation_filter, sol, eachcol(y))
    return y
end

function observation_filter(df::DataFrame)
    z = complex.(df[:x], df[:y])
    return reshape(z, 37, 50)
end

export pre_step_mean, post_step_mean
function pre_step_mean(sol::EnsembleSolution)
    u = timepoint_mean(sol, 0:5:90)
    return u
end
function post_step_mean(sol::EnsembleSolution)
    u = timepoint_mean(sol, 90:5:180)
    return u .- u[1]
end

export pre_step_traj, post_step_traj
function pre_step_traj(sol)
    x = sol.(0:5:90)
end
function post_step_traj(sol)
    x = sol.(90:5:180)
    x .-= x[1]
    x
end