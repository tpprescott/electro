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
    log_importance::Float64
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
    n=500,
    σ,
    kwargs...
) where Names
    
    temperature = zero(Float64)
    gen = zero(Int64)
    
    # Step 0
    B = initInferenceBatch(N, π, L; n=n)

    while true
        # Step 1
        gen += one(Int64)

        Δt, ess = find_dt(B, temperature; kwargs...)
        temperature += Δt
        @info "Generation: $(gen). Temperature: $(temperature). ESS: $(ess)."
        temperature<1 || (return B)

        perturb!(B, L, π, temperature, σ; n=n)
    end
end

function find_dt(
    B::InferenceBatch, 
    temperature;
    alpha=0.6,
    Δt_max=1.0,
    Δt_min=1e-3,
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

function perturb!(B::InferenceBatch, L, π::ParameterDistribution{Names}, temp, σ; n) where Names
    
    # Build the importance distribution
    ℓ = ell(B, temp)
    w = exp.(ℓ .- maximum(ℓ))
    w ./= sum(w)
    q = MixtureModel(map(p -> MvNormal(p.θ.θ, σ), B), w)

    # Sample from the importance distribution
    N = length(B)
    θstar = rand(q, N)
    for j in 1:N
        θstar_j = selectdim(θstar, 2, j)
        while !insupport(π.π, θstar_j)
            θstar_j .= rand(q)
        end
        B.θ[j].θ .= θstar_j
        B.log_importance[j] = logpdf(π.π, θstar_j) - logpdf(q, θstar_j)
    end
        
    # Simulating only with live particles, get the log_sl of the proposed parameters
    B.log_sl .= get_log_sl(L, B.θ, trues(N), n)
    return nothing
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
    (sum(exp, (ell.-ell_max))^2)/sum(exp, (ell.-ell_max).*2)
end
ESS(B::InferenceBatch, t=1.0) = ESS(ell(B, t))


