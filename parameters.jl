export Parameters, ParameterVector, ParameterSet
export Prior, Importance, Sequential
export insupport

# Define typed arrays that correspond to parameter names
struct Parameters{Names, N, T, M<:AbstractArray{T,N}} <: AbstractArray{T, N}
    θ::M
    function Parameters(θ::M, Names::NTuple{Dim, Symbol}) where M<:AbstractArray{T, N} where T<:Real where N where Dim
        size(θ,1)==Dim || error("Parameter names don't match dimension of parameter space.")
        return new{Names, N, T, M}(θ)
    end
end
# Alias for 1D parameter set - i.e. a single parameter vector
ParameterVector{Names, T, M} = Parameters{Names, 1, T, M}

Base.size(P::Parameters) = size(P.θ)
Base.getindex(P::Parameters, i::Int...) = getindex(P.θ, i...)
Base.setindex!(P::Parameters, v, i::Int...) = setindex!(P.θ, v, i...)
Base.hcat(P::Parameters{Names}...) where Names = Parameters(hcat((P_i.θ for P_i in P)...), Names)

# Vectorise the 2D array to be able to run through the parameter vectors individually

struct ParameterSet{Names, T, M<:AbstractMatrix{T}} <: AbstractVector{ParameterVector}
    θ::M
    function ParameterSet(θ::M, Names::NTuple{Dim, Symbol}) where M<:AbstractMatrix{T} where Dim where T<:Real
        size(θ,1)==Dim || error("Parameter names don't match dimension of parameter space.")
        return new{Names, T, M}(θ)
    end
end

Base.size(P::ParameterSet) = (size(P.θ,2),)
Base.getindex(P::ParameterSet{Names}, i::Int) where Names = Parameters(P.θ[:,i], Names)
function Base.setindex!(P::ParameterSet, v, i::Int)
    P.θ[:,i] .= v
end

Parameters(P::ParameterSet{Names}) where Names = Parameters(P.θ, Names)
ParameterSet(P::Parameters{Names, 2}) where Names = ParameterSet(P.θ, Names)
Base.promote_rule(::Type{ParameterSet{Names, T, M}}, ::Type{Parameters{Names, 2, T, M}}) where Names where T where M= Parameters{Names, 2, T, M}

abstract type ParameterDistribution{Names} <: ContinuousMultivariateDistribution end
Base.length(::ParameterDistribution{Names}) where Names = length(Names)
Parameters(π::ParameterDistribution{Names}, n...) where Names = Parameters(rand(π, n...), Names)

struct Prior{Names} <: ParameterDistribution{Names}
    π::ContinuousMultivariateDistribution

    function Prior(I=[])
        issubset(I, [1,2,3,4]) || error("Input subset of [1,2,3,4] only!")
        J = [1,2,3,(I.+3)...]
        π = product_distribution(prior_support[J])
        return new{par_names[J]}(π)
    end
end

Base.eltype(π::Prior) = eltype(π.π)
Distributions._rand!(rng::Random.AbstractRNG, π::Prior, x::AbstractVector{T}) where T<:Real = Distributions._rand!(rng, π.π, x)
Distributions._rand!(rng::Random.AbstractRNG, π::Prior, x::AbstractMatrix{T}) where T<:Real = Distributions._rand!(rng, π.π, x)
function Distributions.insupport(π::Prior{Names}, x::Parameters{Names}) where Names 
    Distributions.insupport(π.π, x)
end
function Distributions.insupport(::Prior{Names}, x::Parameters{DiffNames,N}) where Names where DiffNames where N
    return N==1 ? false : falses(size(x)[2:end])
end

Distributions._logpdf(π::Prior, x::AbstractArray) = Distributions._logpdf(π.π, x)
Distributions._logpdf!(r::AbstractArray, π::Prior, x::AbstractArray{T,2}) where T<:Real = Distributions._logpdf!(r, π.π, x)
Distributions._pdf!(r::AbstractArray, π::Prior, x::AbstractArray{T,2}) where T<:Real = Distributions._pdf!(r, π.π, x)

## Define importance distributions based on resampling from existing parameter samples

struct Importance{Names, MM} <: ParameterDistribution{Names}
    q::MM

    function Importance(X::Parameters{Names, 2}) where Names
        cov_matrix = cov(X, dims=2)
        q = MixtureModel(map(θ->MvNormal(θ, cov_matrix), eachcol(X)))
        MM = typeof(q)
        return new{Names, MM}(q)
    end
    function Importance(X::Parameters{Names, 2}, w::Weights) where Names
        cov_matrix = StatsBase.cov(X.θ, w, 2)
        q = MixtureModel(map(θ->MvNormal(θ, cov_matrix), eachcol(X)), w./w.sum)
        MM = typeof(q)
        return new{Names, MM}(q)
    end
end

Base.eltype(q::Importance) = eltype(q.q)
Distributions._rand!(rng::Random.AbstractRNG, q::Importance, x::AbstractVector{T}) where T<:Real = Distributions._rand!(rng, q.q, x)
Distributions._rand!(rng::Random.AbstractRNG, q::Importance, x::AbstractMatrix{T}) where T<:Real = Distributions._rand!(rng, q.q, x)

Distributions._logpdf(q::Importance, x::AbstractArray) = Distributions._logpdf(q.q, x)
Distributions._logpdf!(r::AbstractArray, q::Importance, x::AbstractArray{T,2}) where T<:Real = Distributions._logpdf!(r, q.q, x)
Distributions._pdf!(r::AbstractArray, q::Importance, x::AbstractArray{T,2}) where T<:Real = Distributions._pdf!(r, q.q, x)

## Sequential inference - half NoEF importance distribution, half prior 
struct Sequential{Names, MM} <: ParameterDistribution{Names}
    NoEF::MM
    EF::ContinuousMultivariateDistribution

    function Sequential(X::Parameters{(:v, :EB_on, :EB_off, :D), 2}, I=[1,2,3,4])
        cov_matrix = cov(X.θ, dims=2)
        NoEF = MixtureModel(map(θ->MvNormal(θ, cov_matrix), eachcol(X)))
        MM = typeof(NoEF)
        
        EF = product_distribution(prior_support[I.+3])

        Names = par_names[[1,2,3,(I.+3)...]]
        return new{Names, MM}(NoEF, EF)
    end
    function Sequential(X::Parameters{(:v, :EB_on, :EB_off, :D), 2}, w::Weights, I=[1,2,3,4])
        cov_matrix = StatsBase.cov(X.θ, w, 2)
        NoEF = MixtureModel(map(θ->MvNormal(θ, cov_matrix), eachcol(X)), w./w.sum)
        MM = typeof(NoEF)
        
        EF = product_distribution(prior_support[I.+3])

        Names = par_names[[1,2,3,(I.+3)...]]
        return new{Names, MM}(NoEF, EF)
    end
end

Base.eltype(q::Sequential) = (eltype(q.NoEF)==eltype(q.EF)) ? eltype(q.NoEF) : error("Components of the Sequential distribution do not match element types")
function Distributions._rand!(rng::Random.AbstractRNG, q::Sequential, x::AbstractVector{T}) where T<:Real
    Distributions._rand!(rng, q.NoEF, view(x, 1:3))
    Distributions._rand!(rng, q.EF, view(x, Not(1:3)))
    x
end
function Distributions._rand!(rng::Random.AbstractRNG, q::Sequential, x::AbstractMatrix{T}) where T<:Real
    Distributions._rand!(rng, q.NoEF, selectdim(x, 1, 1:3))
    Distributions._rand!(rng, q.EF, selectdim(x, 1, Not(1:3)))
    x
end

function Distributions._logpdf(q::Sequential, x::AbstractVector)
    ell = Distributions._logpdf(q.NoEF, view(x, 1:3))
    ell += Distributions._logpdf(q.EF, view(x, Not(1:3)))
    ell
end
function Distributions._logpdf(q::Sequential, x::AbstractMatrix)
    ell = Distributions._logpdf(q.NoEF, selectdim(x, 1, 1:3))
    ell .+= Distributions._logpdf(q.EF, selectdim(x, 1, Not(1:3)))
    ell
end
