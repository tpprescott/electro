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
## Note that StructArrays make particles mutable!

function initInferenceBatch(N::Int, π::ParameterDistribution, L::SyntheticLogLikelihood; n::Int64=500)
    θ = ParameterSet(π, N)
    f(t) = L(t, n=n)
    log_sl = @showprogress pmap(f, θ)
    return StructArray(Particle.(θ, log_sl, Ref(0.0)))
end

#=
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
=#

#=
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
=#

##################
# SMC: Step and wrapper
##################
function smc(
    L::SyntheticLogLikelihood, 
    π::ParameterDistribution{Names}, 
    N::Int;
    N_T::Int=N,
    synthetic_likelihood_n=500,
    σ,
    resample_factor = 1.0,
    expand_factor = 1.0,
    kwargs...
) where Names
    
    temperature = zero(Float64)
    gen = zero(Int64)
    
    # Step 0
    B = initInferenceBatch(N, π, L; n=synthetic_likelihood_n)

    while true
        # Step 1
        gen += one(Int64)
        temperature<1 || (return B)

        Δt, ess = find_dt(B, temperature; kwargs...)
        temperature += Δt
        @info "Generation: $(gen). Temperature: $(temperature). ESS: $(ess)."

        if ess<N_T
            resample!(B)
            σ ./= resample_factor
        else
            σ .*= expand_factor
        end
        K = MvNormal(σ)
        n = perturb!(B, L, π, temperature, K; synthetic_likelihood_n=synthetic_likelihood_n)
        while n<N
            @info "$n unique parameter values: perturbing again"
            n = perturb!(B, L, π, temperature, K; synthetic_likelihood_n=synthetic_likelihood_n)
        end
    end
end

function find_dt(
    B::InferenceBatch, 
    temperature;
    alpha=0.8,
    Δt_max=1.0,
    Δt_min=1e-6,
)

    0<=temperature<1 || error("Temperature must be between zero and 1") 
    
    # Find the change in temperature to reduce ESS by factor of alpha<1
    f(Δtemp) = alpha*ESS(B) - tryESS(B, Δtemp)
    
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

    B.ell .+= Δt.*B.log_sl
    B.ell .-= maximum(B.ell)
    Δt, ESS(B)
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
    return length(unique(B.θ))
end
