module SimpleElectro

using DifferentialEquations
using Roots
using LinearAlgebra, Distances
using Distributions, StatsBase
using RecipesBase
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

export EF, ElectroSim, Parameters, ParDistribution
export SyntheticLogLikelihood
export ESS
export mcmc!, smc

struct EF
    u0::Vector{Float64}
    switches::Dict{Float64, Vector{Float64}}
    function EF(switches=Dict{Float64, Vector{Float64}}(); u0=zeros(2))
        for u in values(switches)
            length(u)==2 || error("Need 2D u")
        end
        return new(u0, switches)
    end
end
function EF(ts, us; u0=zeros(2))
    switches = Dict(Float64(ti)=>Float64.(ui) for (ti, ui) in zip(ts, us))
    return EF(switches; u0=u0)
end
Base.length(U::EF) = length(U.switches)

struct ElectroSim{T}
    sol::T
    function ElectroSim(args...; dt=2^-8, σ0=1.0, kwargs...)
        prob = makeProb(args...; σ0=σ0, kwargs...)
        sol = solve(prob, adaptive=false, dt=dt, saveat=5, save_idxs=[1,2])
        T = typeof(sol)
        return new{T}(sol)
    end
end
@recipe function f(sim::ElectroSim{T}) where T 
    aspect_ratio := :equal
    framestyle --> :origin
    legend --> :none
    xticks --> []
    yticks --> []
    (sim.sol[1,:], sim.sol[2,:])
end

struct Parameters{N} <: AbstractArray{NamedTuple{N}, 1}
    θ::Matrix{Float64}
    logW::Vector{Float64}
    function Parameters{N}(θ) where N
        d, n = size(θ)
        d==length(N) || error("Names length must equal dimension of each parameter vector")
        return new{N}(θ, zeros(n))
    end
end
Parameters{N}(D::MultivariateDistribution; n::Int=1) where N = Parameters{N}(rand(D, n))
Parameters{N}(intervals...; n::Int=1) where N = Parameters{N}(product_distribution([Uniform(I...) for I in intervals]); n=n)

Base.length(p::Parameters) = size(p.θ, 2)
Base.size(p::Parameters) = (length(p),)
Base.IndexStyle(::Parameters) = IndexLinear()
Base.getindex(p::Parameters{N}, i::Integer) where N = NamedTuple{N}(p.θ[:,i])
Base.getindex(p::Parameters{N}, i) where N = NamedTuple{N}.(eachcol(p.θ[:,i]))
function Base.setindex!(p::Parameters, v, i)
    p.θ[:,i] .= v
end
Base.firstindex(p::Parameters) = 1
Base.lastindex(p::Parameters) = length(p)

sq(x) = x^2
function ESS(p::Parameters)
    p.logW .-= maximum(p.logW)
    return ESS(p.logW)
end
function ESS(logW::AbstractVector{Float64})
    M = maximum(logW)
    w = broadcast(exp, logW.-M)
    num = sq(sum(w))
    den = sum(sq, w)
    return num/den
end

function ElectroSim(θ::Parameters, args...; kwargs...)
    simVec = map(θ) do p
        ElectroSim(args...; kwargs..., p...)
    end
    return simVec
end
function ElectroSim(θ::Parameters, n::Integer, args...; kwargs...)
    simMat = map(Iterators.product(θ, 1:n)) do (p,i)
        ElectroSim(args...; kwargs..., p...)
    end
    return simMat
end

struct ParDistribution{N, D<:MultivariateDistribution}
    π::D
    function ParDistribution{N}(π::D) where N where D<:MultivariateDistribution
        length(π)==length(N) || error("Distribution dimension does not match names tuple $N")
        return new{N,D}(π)
    end
end
function ParDistribution{N}(intervals...) where N
    π = product_distribution([Uniform(I...) for I in intervals])
    return ParDistribution{N}(π)
end
function ParDistribution(θ::Parameters{N}; Σ=cov(θ.θ')) where N
    W = exp.(θ.logW)
    W ./= sum(W)
    π = MixtureModel(map(t->MvNormal(t, Σ), eachcol(θ.θ)), W)
    return ParDistribution{N}(π)
end

Parameters(pd::ParDistribution{N}; n::Int=1) where N = Parameters{N}(pd.π; n=n)
Distributions.insupport(::ParDistribution, ::Parameters) = false
Distributions.insupport(pd::ParDistribution{N}, x::Parameters{N}) where N = insupport(pd.π, x.θ)
Distributions.logpdf(::ParDistribution, ::Parameters) = -Inf
Distributions.logpdf(pd::ParDistribution{N}, x::Parameters{N}) where N = logpdf(pd.π, x.θ)

function Proposals(K::MultivariateDistribution)
    iszero(mean(K)) || error("Proposal distribution needs zero mean")
    f = function(θ::Parameters{N}) where N
        n = length(θ)
        dθ = rand(K, n)
        dθ .+= θ.θ
        θstar = Parameters{N}(dθ)
        θstar.logW .= θ.logW
        return θstar
    end
    return f
end
Proposals(Σ::Array{Float64}) = Proposals(MvNormal(Σ))
Proposals(θ::Parameters) = Proposals(cov(θ.θ'))
function resample(θ::Parameters{N}) where N
    n = length(θ)
    w = Weights(exp.(θ.logW))
    idx = sample(1:n, w, n)
    return Parameters{N}(θ.θ[:,idx])
end

function mcmc!(θ::Parameters{N}, prior::ParDistribution{N}, L...; t=1, proposal_dist = Proposals(θ)) where N
    n = length(θ)
    θstar = proposal_dist(θ)
    α = logpdf(prior, θstar)
    α .-= logpdf(prior, θ)

    logLstar = zeros(n)
    logL = zeros(n)
    for Lᵢ in L
        logLstar .+= Lᵢ(θstar)
        logL .+= Lᵢ(θ)
    end

    @. α += t*(logLstar - logL)
    u = log.(rand(n))
    accepted = u.<α

    view(θ.θ, :, accepted) .= view(θstar.θ, :, accepted)
    view(logL, accepted) .= view(logLstar, accepted)
    acceptance_rate = count(accepted)/n
    @info "MCMC step acceptance rate: $acceptance_rate"
    return logL
end
function mcmc!(θ::Parameters, n::Integer, args...; kwargs...)
    logL = zeros(length(θ))
    for i in 1:n
        logL .= mcmc!(θ, args...; kwargs...)
    end
    return logL
end

function _tempIncrement(θ::Parameters, logL, λ, t, dt_range)
    dt_0, dt_1 = dt_range
    residual(dt) = ESS(θ.logW .+ dt*logL) - λ*ESS(θ)

    dt_min = min(dt_0, 1-t)
    dt_max = min(dt_1, 1-t)
    
    r0 = residual(dt_min)
    r1 = residual(dt_max)

    dt = if r0≤0
        dt_min
    elseif r1≥0
        dt_max
    else
        fzero(residual, dt_min, dt_max)
    end
    return dt
end

function smc(
    prior::ParDistribution, 
    L...;
    Σ0,
    n=1000,
    dt_range=(1e-6, 1.0),
    λ=7/8,
    ResampleESS=n*(λ^5),
)

    θ = Parameters(prior, n=n)
    prop_dist = Proposals(Σ0)

    logL = zeros(n)
    for Lᵢ in L
        logL .+= Lᵢ(θ)
    end
    t = 0.0
    while t < 1.0
        dt = _tempIncrement(θ, logL, λ, t, dt_range)
        t += dt
        θ.logW .+= dt*logL
        ess = ESS(θ)
        @info "temperature = $t ; ESS = $ess"
        if ess ≤ ResampleESS
            θ = resample(θ)
            prop_dist = Proposals(θ)
        end
        logL .= mcmc!(θ, prior, L...; t=t, proposal_dist=prop_dist)
    end
    return θ
end
    
# Simulation functions
function f(dx, x, p, t)
    
    dpos = view(dx, 1:2)
    dpol = view(dx, 3:4)
    pos = view(x, 1:2)
    pol = view(x, 3:4)
    
    γ₁ = get(p, :γ₁, 0.0)
    γ₂ = get(p, :γ₂, 0.0)
    γ₃ = get(p, :γ₃, 0.0)
    γ₄ = get(p, :γ₄, 0.0)
    u = get(p, :u, [0.0,0.0])
    v = p[:v]
    D = abs(p[:D])

    npol = norm(pol)
    nu = norm(u)

    @. dpol = -D*(pol - γ₄*u)
    @. dpos = γ₁*v*u
    if !iszero(npol)
        @. dpos += v*pol*(1 + γ₂*nu + γ₃*u*pol/npol)
    end
    return nothing
end
function g(dx, x, p, t)
    σ = sqrt(2*abs(p[:D]))
    dx[1] = 0.0
    dx[2] = 0.0
    dx[3] = σ
    dx[4] = σ
end

function makeSwitches(U::EF)
    condition(u, t, integrator) = t ∈ keys(U.switches)
    function affect!(integrator)
        t_switch = integrator.t
        integrator.p[:u] .= U.switches[t_switch]
    end
    cb = DiscreteCallback(condition, affect!, save_positions=(false, true))
end

function makeProb(U::EF = EF(), tspan = (0.0,300.0); σ0=1.0, kwargs...)
    pol0=σ0.*randn(2)
    _makeProb(U, tspan, pol0; u=U.u0, kwargs...)
end
function _makeProb(U::EF, tspan, pol0; kwargs...)
    return SDEProblem(
        f,
        g,
        vcat(zeros(2), pol0),
        tspan,
        kwargs;
        callback=makeSwitches(U),
        tstops=collect(keys(U.switches)),
    )
end

function initialiser(parameterVectors::Parameters{N}, U::EF = EF(), tspan = (0.0, 300.0); batch_size::Integer, σ0=1.0, kwargs...) where N
    f = function (prob, i, repeat)
        n = cld(i, batch_size)
        pars = parameterVectors[n]
        makeProb(U, tspan; σ0=σ0, prob.p..., kwargs..., pars...)
    end
    return f
end
function initialiser(U::EF = EF(), tspan = (0.0, 300.0); σ0=1.0, kwargs...)
    f = function (prob, i, repeat)
        makeProb(U, tspan; σ0=σ0, prob.p..., kwargs...)
    end
    return f
end

function _summarise!(y, sol)
    n = size(sol, 2)
    y[1] = sol[1, n] - sol[1, 1]
    y[2] = euclidean(sol[:, 1], sol[:, n])
    x0 = copy(sol[:, 1])
    L, L2 = zeros(2)
    for x1 in Iterators.drop(eachcol(sol), 1)
        L += euclidean(x0, x1)
        L2 += sqeuclidean(x0, x1)
        x0 .= x1
    end
    y[3] = L
    y[4] = sqrt((L2/(n-1)) - ((L/(n-1))^2))
    return y
end

function summariser()
    f = function(sol, i)
        y = zeros(4)
        _summarise!(y, sol)
        return y, false
    end
    return f
end
function summariser(idx_range)
    f = function(sol, i)
        y = zeros(4)
        _summarise!(y, sol[idx_range])
        return y, false
    end
    return f
end
function summariser(idx_ranges...)
    f = function(sol, i)
        M = length(idx_ranges)
        y = zeros(4, M)
        
        for (idx_range, yᵢ) in zip(idx_ranges, eachcol(y))
            _summarise!(yᵢ, sol[idx_range])
        end
        return vec(y), false
    end
    return f
end
function summariser(U::EF, t_span)
    t0, tf = t_span
    I0 = 1+Int(t0/5)
    If = 1+Int(tf/5)
    if length(U)==0
        return summariser(I0:If)
    else
        tstops = sort(collect(keys(U.switches)))
        idxstops = @. 1 + Int.(tstops./5)

        i_from = [I0]
        i_to = Int64[]
        for k in idxstops
            if I0 < k < If
                push!(i_to, k)
                push!(i_from, k)
            end
        end
        push!(i_to, If)
        return summariser((a:b for (a,b) in zip(i_from, i_to))...)
    end
end

function batchSummaryStats()
    f = function(u, data, I)
        u = push!(u, hcat(data...))
        u, false
    end
    u_init = Array{Float64, 2}[]
    return (reduction=f, u_init=u_init)
end
function batchEmpiricalDistribution(::Type{D}=MvNormal) where D<:MultivariateDistribution
    f = function(u, data, I)
        X = fit(D, hcat(data...))
        u = push!(u, X)
        u, false
    end
    u_init = D[]
    return (reduction=f, u_init=u_init)
end
function batchSyntheticLogLikelihood(yobs, ::Type{D}=MvNormal) where D<:MultivariateDistribution
    f = function(u, data, I)
        X = fit(D, hcat(data...))
        ℓ = logpdf(X, yobs)
        u = push!(u, sum(ℓ))
        u, false
    end
    u_init = Float64[]
    return (reduction=f, u_init=u_init)
end

function SyntheticLogLikelihood(n::Int=500, U::EF=EF(), tspan=(0.0,300.0); data, kwargs...)
    f = function(θ::Parameters)
        P = makeProb(U, tspan; θ[1]..., kwargs...)
        EP = EnsembleProblem(
            P;
            prob_func=initialiser(θ, batch_size=n),
            output_func=summariser(U, tspan),
            batchSyntheticLogLikelihood(data)...)
    
        es = solve(EP, SOSRA(), EnsembleDistributed(), save_idxs=[1,2], saveat=5, trajectories=length(θ)*n, batch_size=n, progress=true)
        return es.u
    end
    return f
end

using CSV, DataFrames
function _readObservations(fn::String)
    df = CSV.read(fn, DataFrame)
    traj = groupby(df, :TID)
    return traj
end


export ElectroData
struct ElectroData
    xy::Array{Float64,2}
end
@recipe function f(D::ElectroData)
    aspect_ratio := :equal
    framestyle --> :origin
    legend --> :none
    xticks --> []
    yticks --> []
    (selectdim(D.xy,2,1), selectdim(D.xy,2,2))
end
@recipe function f(Dvec::AbstractVector{ElectroData})
    aspect_ratio := :equal
    framestyle --> :origin
    legend --> :none
    xticks --> []
    yticks --> []
    x = hcat((selectdim(D.xy,2,1) for D in Dvec)...)
    y = hcat((selectdim(D.xy,2,2) for D in Dvec)...)
    (x, y)
end

function getPositions()
    f = function(tr)
        X = Array(tr[!, [:x,:y]])
        n = size(X,1)
        selectdim(X, 1, 1:n) .-= selectdim(X, 1, 1:1)
        return ElectroData(X)
    end
    return f
end
function ElectroData(fn::String)
    traj = _readObservations(fn)
    f = getPositions()
    y_vec = map(f, traj)
    return y_vec
end

export xobs_Ctrl, xobs_Ctrl_1, xobs_Ctrl_2
const xobs_Ctrl_1 = ElectroData("data/test/Ctrl-1.csv")
const xobs_Ctrl_2 = ElectroData("data/test/Ctrl-2.csv")
const xobs_Ctrl = vcat(xobs_Ctrl_1, xobs_Ctrl_2)
export xobs_200, xobs_200_1, xobs_200_2
const xobs_200_1 = ElectroData("data/test/200mV-1.csv")
const xobs_200_2 = ElectroData("data/test/200mV-2.csv")
const xobs_200 = vcat(xobs_200_1, xobs_200_2)


function getSummaries()
    f = function(tr)
        y = zeros(4)
        X = Array(tr[!, [:x, :y]])
        _summarise!(y, permutedims(X))
        return y
    end
    return f
end
function getSummaries(idx_ranges...)
    f = function(tr)
        M = length(idx_ranges)
        y = zeros(4, M)
        X = Array(tr[!, [:x, :y]])
        for (k,I) in enumerate(idx_ranges)
            X_k = selectdim(X, 1, I)
            y_k = selectdim(y, 2, k)
            _summarise!(y_k, permutedims(X_k))
        end
        return vec(y)
    end
    return f
end

export ElectroSummaries
function ElectroSummaries(fn::String, idx_ranges...)
    traj = _readObservations(fn)
    f = getSummaries(idx_ranges...)
    yvec = map(f, traj)
    return hcat(yvec...)
end

export yobs_Ctrl, yobs_Ctrl_1, yobs_Ctrl_2
const yobs_Ctrl_1 = ElectroSummaries("data/test/Ctrl-1.csv")
const yobs_Ctrl_2 = ElectroSummaries("data/test/Ctrl-2.csv")
const yobs_Ctrl = hcat(yobs_Ctrl_1, yobs_Ctrl_2)
export yobs_200, yobs_200_1, yobs_200_2
const yobs_200_1 = ElectroSummaries("data/test/200mV-1.csv", 1:13, 13:37, 37:61)
const yobs_200_2 = ElectroSummaries("data/test/200mV-2.csv", 1:13, 13:37, 37:61)
const yobs_200 = hcat(yobs_200_1, yobs_200_2)

export U_Ctrl, U_200
const U_Ctrl = EF()
const U_200 = EF([60,180],[[1,0],[-1,0]])

export L_Ctrl, L_Ctrl_1, L_Ctrl_2
const L_Ctrl_1 = SyntheticLogLikelihood(500, U_Ctrl, (0.0,300.0), data=yobs_Ctrl_1)
const L_Ctrl_2 = SyntheticLogLikelihood(500, U_Ctrl, (0.0,300.0), data=yobs_Ctrl_2)
const L_Ctrl = SyntheticLogLikelihood(500, U_Ctrl, (0.0,300.0), data=yobs_Ctrl)

export L_train_200, L_train_200_1, L_train_200_2
const L_train_200_1 = SyntheticLogLikelihood(500, U_200, (0.0,180.0), data=yobs_200_1[1:8,:])
const L_train_200_2 = SyntheticLogLikelihood(500, U_200, (0.0,180.0), data=yobs_200_2[1:8,:])
const L_train_200 = SyntheticLogLikelihood(500, U_200, (0.0,180.0), data=yobs_200[1:8,:])

end

