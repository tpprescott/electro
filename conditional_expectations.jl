function Distributions.mean(Z::TrajectoryRandomVariable, n::Int64)
    y = rand(Z, n)
    return mean(y, dims=2)
end
function Distributions.mean(Z::TrajectoryRandomVariable, y)
    Distributions.rand!(Z, y)
    return mean(y, dims=2)
end
