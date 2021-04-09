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
    copies::Int64
end
InferenceBatch{P} = StructArray{Particle{P}}
## Note that StructArrays make particles mutable!

function initInferenceBatch(N::Int, π::ParameterDistribution, L::SyntheticLogLikelihood; n::Int64=500)
    θ = ParameterSet(π, N)
    f(t) = L(t, n=n)
    log_sl = @showprogress pmap(f, θ)
    return StructArray(Particle.(θ, log_sl, Ref(0.0), Ref(1)))
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
    kwargs...
) where Names
    
    K = MvNormal(σ)
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

        B = ess<N_T ? resample(B) : B
        B = perturb(B, L, π, temperature, K; synthetic_likelihood_n=synthetic_likelihood_n)
    end
end

function find_dt(
    B::InferenceBatch, 
    temperature;
    alpha=0.8,
    Δt_min=1e-6,
)

    0<=temperature<1 || error("Temperature must be between zero and 1") 
    
    # Find the change in temperature to reduce ESS by factor of alpha<1
    f(Δtemp) = alpha*ESS(B) - tryESS(B, Δtemp)
    
    Δt_max = min(0.1, 1-temperature)

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
function get_log_sl(L::SyntheticLogLikelihood, θ::ParameterSet, isalive, n::Int)
    f(a,θ_i) = a ? L(θ_i, n=n) : -Inf
    log_sl = @showprogress pmap(f, isalive, θ)
end

# The other functions
function ESS(scaled_ell::Vector{Float64}) 
    scaled_ell .-= maximum(scaled_ell)
    sum(exp, scaled_ell)^2/sum(exp, scaled_ell.*2)
end

ESS(B::InferenceBatch) = ESS(B.ell, B.copies)
ESS(ell::Vector{Float64}, copies::Vector{Int64}) = ESS(ell .+ log.(copies))
tryESS(B::InferenceBatch, Δtemp) = tryESS(B.ell, B.copies, B.log_sl, Δtemp)
tryESS(ell::Vector{Float64}, copies::Vector{Int64}, log_sl::Vector{Float64}, Δtemp) = ESS(ell .+ log.(copies) .+ (Δtemp.*log_sl))

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

function resample(B::InferenceBatch)
    n = length(B)
    N = sum(B.copies)
        
    W = Weights(B.copies .* exp.(B.ell))
    I = sample(1:n, W, N)
    
    B.copies .= 0
    B.ell .= 0.0
    for i in I
        B.copies[i] += 1
    end
    return filter(checkvalid, B)
end

function perturb(B::InferenceBatch, L, π::ParameterDistribution{Names}, temp, K::MvNormal; synthetic_likelihood_n) where Names
    n = length(B)
    N = sum(B.copies)

    # Set of perturbed parameter values
    θstar_mat = rand(K, N)
    cidx=0
    for i in 1:n
        for j in 1:B.copies[i]
            cidx+=1
            θstar_mat[:, cidx] .+= B.θ[i].θ
        end
    end
    θstar = ParameterSet(θstar_mat, Names)
    
    # Simulating only with live particles, get the log_sl of the proposed parameters
    isalive = insupport.(Ref(π), θstar)
    log_sl_star = get_log_sl(L, θstar, isalive, synthetic_likelihood_n)

    # NOTE - TWO LOOPS TO ALLOW PARALLELISATION IN THE MIDDLE

    # Accept perturbations according to an M-H acceptance kernel
    cidx=0
    for i in 1:n
        J = B.copies[i]
        for j in 1:J
            cidx+=1
            log_α = temp*(log_sl_star[cidx] - B.log_sl[i]) + logpdf(π, θstar[cidx]) - logpdf(π, B.θ[i])
            if log(rand()) < log_α
                B.copies[i] -= 1
                p = Particle(θstar[cidx], log_sl_star[cidx], B.ell[i], 1)
                push!(B, p)
            end
        end
    end
    return filter(checkvalid, B)
end
