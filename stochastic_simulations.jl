export AbstractPolarityDistribution
export RandomInitialPolarity
export TrajectoryDistribution
export TrajectoryRandomVariable

### PROB FUNC(S) (prob, i, repeat) -> 

abstract type AbstractPolarityDistribution
end
function prob_func(p0::AbstractPolarityDistribution)
    pf = function (base, i, repeat)
        remake(base, u0=p0())
    end
    return pf
end

struct RandomInitialPolarity{T} <: AbstractPolarityDistribution
    σ::T
end
(r::RandomInitialPolarity)() = [r.σ*randn(ComplexF64), zero(ComplexF64)]

struct FixedInitialPolarity <: AbstractPolarityDistribution
    p0::ComplexF64
end
(r::FixedInitialPolarity)() = [r.p0, zero(ComplexF64)]

### BUILD A MONTE CARLO SAMPLE OF SUMMARY STATISTICS FOR A GIVEN PARAMETER, POLARITY DISTRIBUTION, AND EMF

include("sde.jl")
struct TrajectoryDistribution{B<:SDEProblem, E<:EnsembleProblem}
    base::B
    prob::E
end
function TrajectoryDistribution(
    θ::Parameters{pars,1}, 
    p0::AbstractPolarityDistribution, 
    u::AbstractEMField;
    tspan=(0.0, 180.0),
) where pars

    base = SDEProblem(
        SDEdrift(u; NamedTuple{pars}(θ)...), 
        SDEnoise(; NamedTuple{pars}(θ)...),
        p0(),
        tspan,
        noise_rate_prototype=NOISEFORM,
    )

    prob = EnsembleProblem(
        base,
        prob_func = prob_func(p0),
    )

    return TrajectoryDistribution(base, prob)
end

Base.rand(P::TrajectoryDistribution; kwargs...) = solve(P.base; kwargs...)
Base.rand(P::TrajectoryDistribution, n::Int; kwargs...) = solve(P.prob, trajectories=n; kwargs...)

struct TrajectoryRandomVariable{YY<:AbstractSummary, PP<:TrajectoryDistribution} <: Sampleable{Multivariate, Continuous}
    Y::YY # Summary statistic definition
    P::PP # Trajectory distribution
end
Base.length(Z::TrajectoryRandomVariable) = length(Z.Y)
get_options(Z::TrajectoryRandomVariable) = get_options(Z.Y)

function Distributions._rand!(rng::AbstractRNG, Z::TrajectoryRandomVariable, y::AbstractVector{T}; kwargs...) where T<:Real
    sol = rand(Z.P; get_options(Z)..., kwargs...)
    summarise!(y, sol, Z.Y)
    y
end
function Distributions._rand!(rng::AbstractRNG, Z::TrajectoryRandomVariable, y::DenseMatrix{T}; kwargs...) where T<:Real
    ens_sol = rand(Z.P, size(y,2); get_options(Z)..., kwargs...)
    for (y_i, sol) in zip(eachcol(y), ens_sol)
        summarise!(y_i, sol, Z.Y)
    end
    y
end


