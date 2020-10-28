
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