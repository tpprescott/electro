# struct TrajectoryRandomVariable{YY<:AbstractSummary, PP<:TrajectoryDistribution} <: Sampleable{Multivariate, Continuous}
#     Y::YY # Summary statistic definition
#     P::PP # Trajectory distribution
# end
# struct TrajectoryDistribution{B<:SDEProblem, E<:EnsembleProblem}
#     base::B
#     prob::E
# end

export EmpiricalSummary
export ConditionalExpectation

abstract type EmpiricalSummary end

function Distributions.mean(Z::TrajectoryRandomVariable...; n::Int64)
    dim_y = sum(length, Z)
    y = zeros(dim_y, n)
    mean(y, Z...)
end
function Distributions.mean(y, Z::TrajectoryRandomVariable...)
    d_i=0
    for Z_i in Z
        l = length(Z_i)
        J = d_i .+ (1:l)
        Distributions.rand!(Z_i, selectdim(y, 1, J))
        d_i += l
    end
    return vec(mean(y, dims=2))
end

struct ConditionalExpectation
    D::Matrix{Float64}
    ell::Vector{Float64}
    function ConditionalExpectation(D, ell)
        size(D,2)==length(ell) || error("Mismatched sizes")
        new(D, ell)
    end
end

function ConditionalExpectation(θ::ParameterSet, ell, Φ::EmpiricalSummary; n=500)
    D = zeros(length(Φ), length(θ))
    for (i, θ_i) in enumerate(θ)
        D[:,i] .= Φ(θ_i; n=n)
    end
    return ConditionalExpectation(D, ell)
end