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
end
function InferenceBatch(N::Int, π::ParameterDistribution)
    θ = Parameters(π, N)
    ell = fill(1/N, N)
    log_sl = zeros(N)
    InferenceBatch(θ, ell, log_sl)
end
function InferenceBatch(
    π::ParameterDistribution,
    B::InferenceBatch{par_names_NoEF},
    σ
)

    N = length(B)
    resample!(B)
    K = MvNormal(σ)
    perturb!(B, K)

    θ = Parameters(π, N)
    θ.θ[1:3, :] .= B.θ.θ

    ell = fill(1/N, N)
    log_sl = zeros(N)
    InferenceBatch(θ, ell, log_sl)
end

Base.length(b::InferenceBatch) = length(b.ell)
function Base.append!(B1::InferenceBatch, B2::InferenceBatch)
    B1.θ = hcat(B1.θ, B2.θ)
    append!(B1.ell, B2.ell)
    B1
end
function Base.getindex(b::InferenceBatch{Names}, I) where Names
    Parameters(selectdim(b.θ.θ, 2, I), Names)
end
function Distributions.mean(b::InferenceBatch{Names}) where Names
    b.ell .-= maximum(b.ell)
    w = Weights(exp.(b.ell))
    θbar = vec(mean(b.θ, w, dims=2))
    return Parameters(θbar, Names)
end

function Importance(b::InferenceBatch{Names}) where Names
    b.ell .-= maximum(b.ell)
    idx = isfinite.(b.ell)
    
    w = Weights(exp.(b.ell[idx]))
    Importance(Parameters(b.θ[:,idx], Names), w)
end
function Sequential(b::InferenceBatch{Names}, I...) where Names
    b.ell .-= maximum(b.ell)
    idx = isfinite.(b.ell)

    w = Weights(exp.(b.ell[idx]))
    Sequential(Parameters(b.θ[:,idx], Names), I...)
end
function ConditionalExpectation(B::InferenceBatch, Φ::EmpiricalSummary; n=500)
    idx = isfinite.(B.ell)
    return ConditionalExpectation(ParameterSet(B[idx]), B.ell[idx], Φ; n=n)
end

##################
# SMC: Step and wrapper
##################
function smc(
    L::SyntheticLogLikelihood, 
    π::ParameterDistribution{Names}, 
    N::Int,
    B::InferenceBatch = InferenceBatch(N, π);
    N_T::Int,
    σ,
    kwargs...
) where Names
    
    temperature = zero(Float64)
    gen = zero(Int64)

#    dim = length(Names)
    K = MvNormal(σ)

    while true
        gen += one(Int64)
        Δt, ess = smc_step(B, L, π, temperature; kwargs...)
        temperature += Δt
        @info "Generation: $(gen). Temperature: $(temperature). ESS: $(ess)."

        if temperature >= 1
            return B
        end

        W = Weights(exp.(B.ell))
        ess<N_T && resample!(B)
        perturb!(B, K)
    end
end

function smc_step(
    B::InferenceBatch, 
    L::SyntheticLogLikelihood, 
    π::ParameterDistribution,
    temperature;
    alpha=0.9,
    synthetic_likelihood_n=500,
    Δt_min=1e-5,
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

    Δt = if Δt_max <= Δt_min
        Δt_max
    elseif f(Δt_max) <= 0
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

function _interpolate_covariance!(cov, cov_1, cov_2, λ)
    @. cov = (1-λ)*cov_1 + λ*cov_2
end


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
