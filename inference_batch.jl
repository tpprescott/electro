export InferenceBatch
export isaccepted
export loglikelihood

mutable struct InferenceBatch{Names, P<:Parameters{Names,2}}
    θ::P
    log_π::Vector{Float64}
    log_q::Vector{Float64}
    log_sl::Vector{Float64}
    log_u::Vector{Float64}
end
Base.length(b::InferenceBatch) = length(b.log_π)
function Base.append!(B1::InferenceBatch, B2::InferenceBatch)
    B1.θ = hcat(B1.θ, B2.θ)
    append!(B1.log_π, B2.log_π)
    append!(B1.log_q, B2.log_q)
    append!(B1.log_sl, B2.log_sl)
    append!(B1.log_u, B2.log_u)
    B1
end
function Base.getindex(b::InferenceBatch{Names}, I) where Names
    Parameters(selectdim(b.θ, 2, I), Names)
end

isaccepted(ell::Float64, log_u::Float64, max_ell::Float64) = log_u < ell-max_ell 
function isaccepted(b::InferenceBatch; temp::Real=1.0)
    0 <= temp <= 1 || error("Tempering between 0 and 1 inclusive, 1 for no tempering")
    ell = temp.*b.log_sl .+ b.log_π .- b.log_q
    ell_max = maximum(ell)
    broadcast(isaccepted, ell, b.log_u, Ref(ell_max))
end

# Initialise empty batch
function InferenceBatch(π::ParameterDistribution)
    P = Parameters(π, 0)
    B0 = InferenceBatch(P, Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}())
end

# Produce new batches
function InferenceBatch(
    L::SyntheticLogLikelihood,
    π::ParameterDistribution{Names}, 
    args...;
    kwargs...
) where Names

    B0 = InferenceBatch(π)
    InferenceBatch(B0, L, π, args...; kwargs...)
end

# Increment a batch by N
function InferenceBatch(
    B::InferenceBatch,
    L::SyntheticLogLikelihood,
    π::ParameterDistribution{Names}, 
    q::ParameterDistribution{Names}, 
    N::Int=0;
    synthetic_likelihood_n::Int=500,
) where Names

    P = Parameters(q, N)
    B.θ = hcat(B.θ, P)

    append!(B.log_π, logpdf(π, P))
    append!(B.log_q, logpdf(q, P))
    append!(B.log_u, log.(rand(N)))

    I = insupport(π, P)
    function f(i,θ)::Float64
        i ? L(θ, n=synthetic_likelihood_n) : -Inf
    end
    
    log_sl_inc::Vector{Float64} = @showprogress pmap(f, I, ParameterSet(P))
    append!(B.log_sl, log_sl_inc)
    
    B
end

function InferenceBatch(B::InferenceBatch, L::SyntheticLogLikelihood, π::ParameterDistribution, N::Int=0; synthetic_likelihood_n::Int=500)
    InferenceBatch(B, L, π, π, N; synthetic_likelihood_n=synthetic_likelihood_n)
end

function Distributions.loglikelihood(B::InferenceBatch)
    ell = B.log_sl + B.log_π - B.log_q
    max_ell = maximum(ell)
    return max_ell + log(mean(exp, ell.-max_ell))
end

