export InferenceBatch
export isaccepted
export loglikelihood

export get_log_sl, ESS
export resample!, perturb!
export smc_step, smc

mutable struct InferenceBatch{Names, P<:Parameters{Names,2}}
    θ::P
    ell::Vector{Float64}
    log_sl::Vector{Float64}
    function InferenceBatch(N::Int, π::ParameterDistribution{Names}) where Names 
        θ = Parameters(π, N)
        P = typeof(θ)
        ell = fill(1/N, N)
        log_sl = zeros(N)
        new{Names, P}(θ, ell, log_sl)
    end
end
Base.length(b::InferenceBatch) = length(b.ell)
function Base.append!(B1::InferenceBatch, B2::InferenceBatch)
    B1.θ = hcat(B1.θ, B2.θ)
    append!(B1.ell, B2.ell)
    B1
end
function Base.getindex(b::InferenceBatch{Names}, I) where Names
    Parameters(selectdim(b.θ, 2, I), Names)
end

##################
# SMC: Step and wrapper
##################
function smc(L::SyntheticLogLikelihood, π::ParameterDistribution, N::Int, K::MvNormal; N_T::Int, kwargs...)
    B = InferenceBatch(N, π)
    temperature = Float64(0)
    gen=0
    while true
        get +=1 
        Δt, ess = smc_step(B, L, π, temperature; kwargs...)
        temperature += Δt
        @info "Generation: $(gen). Temperature: $(temperature). ESS: $(ess)."
        
        if temperature >= 1
            return B
        end
        ess<N_T ? resample!(B) : perturb!(B, K)
    end
end

function smc_step(
    B::InferenceBatch, 
    L::SyntheticLogLikelihood, 
    π::ParameterDistribution,
    temperature;
    alpha=0.9,
    synthetic_likelihood_n=500,
    Δt_min=0.001,
)

    0<=temperature<1 || error("Temperature must be between zero and 1")
    # Index so only working with live particles 
    isalive = insupport(π, B.θ)

    # Update the log synthetic likelihoods of the current parameters
    B.log_sl .= get_log_sl(L, ParameterSet(B.θ), isalive, synthetic_likelihood_n)
    
    # Find the change in temperature to reduce ESS by factor of alpha<1
    test_ell = similar(B.ell)
    ESS_now = _ESS(B.ell)
    f(Δtemp) = alpha*ESS_now - _ESS(test_ell, B.ell, B.log_sl, Δtemp)
    
    Δt_max = 1-temperature

    Δt = if f(Δt_max) <= 0
        Δt_max
    elseif f(Δt_min) > 0
        Δt_min
    else
        fzero(f, Δt_min, Δt_max)
    end

    Δt, _ESS(B.ell, B.ell, B.log_sl, Δt)
end


##################
# SMC: Helper functions
##################

# The expensive helper function
function get_log_sl(L::SyntheticLogLikelihood, θ::ParameterSet, isalive, n::Int) where Names
    f(a,θ_i) = a ? L(θ_i, n=n) : -Inf
    log_sl = @showprogress pmap(f, isalive, θ)
end

# The other functions
function _ESS(ell)
    ell .-= maximum(ell)
    sum(exp, ell)^2/sum(exp, 2.0 .* ell)
end
function _ESS(ell_next, ell, log_sl, Δtemp)
    @. ell_next = ell + (Δtemp * log_sl)
    _ESS(ell_next)
end
ESS(B::InferenceBatch) = _ESS(B.ell)


function resample!(B::InferenceBatch{Names}) where Names

    N = length(B)
    B.ell .-= maximum(B.ell)

    W = Weights(exp.(B.ell))
    I = sample(1:N, W, N)
    θ_new = B.θ[:,I]
    B.θ = Parameters(θ_new, Names)

    B.ell .= 1/N
    B
end

function perturb!(B, K::MvNormal)
    N = length(B)
    Δθ = rand(K, N)
    B.θ.θ .+= Δθ
    B
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

