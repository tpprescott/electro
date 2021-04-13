export Particle, InferenceBatch
export isaccepted
export loglikelihood

export get_log_sl, ESS
export resample!, perturb!
export smc_step, smc

using StructArrays
struct Particle{P<:Parameters}
    θ::P
    log_sl::Float64
    ell::Float64
end
InferenceBatch{P} = StructArray{Particle{P}}
ell(p::Particle, t=1.0) = p.log_importance + t*p.log_sl
ell(B::InferenceBatch, t=1.0) = ell.(B, Ref(t))
## Note that StructArrays make particles mutable!

function initInferenceBatch(N::Int, π::ParameterDistribution, L::SyntheticLogLikelihood; n::Int64=500)
    θ = ParameterSet(π, N)
    f(t) = L(t, n=n)
    log_sl = @showprogress pmap(f, θ)
    return StructArray(Particle.(θ, log_sl, Ref(0.0)))
end


##################
# SMC: Step and wrapper
##################
function smc(
    L::SyntheticLogLikelihood, 
    π::ParameterDistribution{Names}, 
    N::Int;
    σ,
    N_T::Int=N,
    synthetic_likelihood_n=500,
    resample_factor = 1.0,
    expand_factor = 1.0,
    nBreathe = 1,
    kwargs...
) where Names
    
    temperature = zero(Float64)
    gen = zero(Int64)
    
    # Step 0
    B = initInferenceBatch(N, π, L; n=synthetic_likelihood_n)

    while true
        # Step 1
        gen += one(Int64)

        Δt, ess = find_dt(B, temperature; kwargs...)
        temperature += Δt
        @info "Generation: $(gen). Temperature: $(temperature). ESS: $(ess)."
        temperature<1 || (return B)

        if ess<N_T
            resample!(B)
            σ ./= resample_factor
        else
            σ .*= expand_factor
        end
        K = MvNormal(σ)
        while !allunique(B.θ)
            perturb!(B, L, π, temperature, K; synthetic_likelihood_n=synthetic_likelihood_n)
        end
    end
end

function find_dt(
    B::InferenceBatch, 
    temperature;
    alpha=0.6,
    Δt_max=1.0,
    Δt_min=1e-6,
    kwargs...
)

    0<=temperature<1 || error("Temperature must be between zero and 1") 
    
    # Find the change in temperature to reduce ESS by factor of alpha<1
    f(Δt) = alpha*ESS(B, temperature) - ESS(B, temperature+Δt)
    
    Δt_max = min(Δt_max, 1-temperature)

    Δt = if Δt_max <= Δt_min
        Δt_max
    elseif f(Δt_max) <= 0
        Δt_max
    elseif f(Δt_min) > 0
        Δt_min
    else
        fzero(f, Δt_min, Δt_max)
    end

    Δt, ESS(B, temperature+Δt)
end


##################
# SMC: Helper functions
##################

# The expensive helper function
function get_log_sl(L::SyntheticLogLikelihood, θ, isalive, n::Int)
    f(a,θ_i) = a ? L(θ_i, n=n) : -Inf
    log_sl = @showprogress pmap(f, isalive, θ)
end

# The other functions
function ESS(ell::Vector{Float64}) 
    ell_max = maximum(ell)
    sum(exp, (ell.-ell_max))^2/sum(exp, (ell.-ell_max).*2)
end
ESS(B::InferenceBatch) = ESS(B.ell)

tryESS(B::InferenceBatch, Δtemp) = tryESS(B.ell, B.log_sl, Δtemp)
tryESS(ell::Vector{Float64}, log_sl::Vector{Float64}, Δtemp) = ESS(ell .+ (Δtemp.*log_sl))

function _interpolate_covariance!(cov, cov_1, cov_2, λ)
    @. cov = (1-λ)*cov_1 + λ*cov_2
end

function checkvalid(p::Particle)
    if iszero(p.copies)
        return false
    elseif isinf(p.ell) && p.ell<0
        return false
    end
    return true
end
ESS(B::InferenceBatch, t=1.0) = ESS(ell(B, t))

function resample!(B::InferenceBatch)
    N = length(B)
        
    W = Weights(exp.(B.ell))
    I = sample(1:N, W, N)
    
    B.θ .= B.θ[I]
    B.log_sl .= B.log_sl[I]
    B.ell .= 0.0
    return nothing
end

function _get_θstar(p::Particle, L, π::ParameterDistribution{Names}, K::MvNormal; synthetic_likelihood_n) where Names
    θstar = Parameters(p.θ.θ + rand(K), Names)
    log_sl_star = insupport(π, θstar) ? L(θstar, n=synthetic_likelihood_n) : -Inf
    return Particle(θstar, log_sl_star, p.ell)
end

function perturb!(B::InferenceBatch, L, π::ParameterDistribution{Names}, temp, K::MvNormal; synthetic_likelihood_n) where Names
    N = length(B)

    # Two steps to perturbation: first, resample simulations for new log likelihood for each particle
    # "Accept" this perturbation w.p. = 1
    isalive = [insupport(π, p.θ) for p in B]
    B.log_sl .= get_log_sl(L, B.θ, isalive, synthetic_likelihood_n)

    # Second, produce set of perturbed parameter values
    f(p::Particle) = _get_θstar(p, L, π, K, synthetic_likelihood_n=synthetic_likelihood_n)
    θstar_list = @showprogress pmap(f, B)
    θstar = StructArray(θstar_list)

    # Accept second perturbation according to an M-H acceptance kernel
    log_α = zeros(N)
    for i in 1:N
        log_α[i] = temp*(θstar.log_sl[i] - B.log_sl[i]) + logpdf(π, θstar.θ[i]) - logpdf(π, B.θ[i])
        if log(rand()) < log_α[i]
            B[i] = θstar[i]
        end
    end
    return log_α
end
