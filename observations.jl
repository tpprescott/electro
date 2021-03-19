export observation_filter

function observation_filter(sol::DESolution; ninterval=37)
    y = zeros(ComplexF64, ninterval)
    observation_filter(sol, y)
end
function observation_filter(sol::DESolution, y::AbstractVector{ComplexF64})
    for j in eachindex(y)
        y[j] = sol(5*(j-1))[2]
    end
    return y
end

function observation_filter(sol::EnsembleSolution; ninterval=37)
    y = zeros(ComplexF64, ninterval, length(sol))
    observation_filter(sol, y)
end
function observation_filter(sol::EnsembleSolution, y::AbstractMatrix{ComplexF64})
    map(observation_filter, sol, eachcol(y))
    return y
end

function observation_filter(df::DataFrame; ninterval=37)
    z = complex.(df[:x], df[:y])
    return reshape(z, ninterval, :)
end

function observation_filter(csv::CSV.File; ninterval=37)
    z = complex.(csv.x, csv.y)
    zmat = reshape(z, ninterval, :)
    for j in axes(zmat,2)
        selectdim(zmat, 2, j) .-= zmat[1,j]
    end
    return zmat
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