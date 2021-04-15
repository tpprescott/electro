# struct TrajectoryRandomVariable{YY<:AbstractSummary, PP<:TrajectoryDistribution} <: Sampleable{Multivariate, Continuous}
#     Y::YY # Summary statistic definition
#     P::PP # Trajectory distribution
# end
# struct TrajectoryDistribution{B<:SDEProblem, E<:EnsembleProblem}
#     base::B
#     prob::E
# end

export EmpiricalSummary, getnames
export S_NoEF, S_Switch, S_Stop
export ConditionalExpectation


function Distributions.mean(Z::TrajectoryRandomVariable...; n::Int64)
    dim_y = sum(length, Z)
    y = zeros(dim_y, n)
    mean(y, Z...)
end
function Distributions.mean(y, Z::TrajectoryRandomVariable...)
    d_i=0
    for Z_i in Z
        l = length(Z_i)
        J = d_i .+ (1:l)
        Distributions.rand!(Z_i, selectdim(y, 1, J))
        d_i += l
    end
    return vec(mean(y, dims=2))
end

struct ConditionalExpectation{Names}
    D::Matrix{Float64}
    ell::Vector{Float64}
    function ConditionalExpectation(D, ell, Names::NTuple{N, Symbol}) where N
        size(D,2)==length(ell) || error("Mismatched sizes: sample points and weights")
        size(D,1)==length(Names) || error("Mismatched sizes: sample dimension and names")
        new{Names}(D, ell)
    end
end
function StatsBase.mean(c::ConditionalExpectation{Names}) where Names
    w = Weights(exp.(c.ell .- maximum(c.ell)))
    NamedTuple{Names}(vec(mean(c.D, w, dims=2)))
end
function StatsBase.cov(c::ConditionalExpectation{Names}) where Names
    w = Weights(exp.(c.ell .- maximum(c.ell)))
    cov(c.D, w, 2)
end
function StatsBase.std(c::ConditionalExpectation{Names}) where Names
    w = Weights(exp.(c.ell .- maximum(c.ell)))
    NamedTuple{Names}(std(c.D, w, 2))
end
function StatsBase.median(c::ConditionalExpectation{Names}) where Names
    w = Weights(exp.(c.ell .- maximum(c.ell)))
    dim= size(c.D, 1)
    return NamedTuple{Names}(map(row -> median(row, w), eachrow(c.D)))
end


##### For analysis

pbar2(theta) = 1/3

abstract type EmpiricalSummary end
function getnames(::Type{T}) where T<:EmpiricalSummary
    error("Define the names of the summary statistic.")
end

function ConditionalExpectation(θ, ell, Φ::ES; n=500) where ES<:EmpiricalSummary
    ϕ(θ_i) = Φ(θ_i; n=n)
    Dvec = @showprogress pmap(ϕ, θ)
    D = hcat(Dvec...)
    return ConditionalExpectation(D, ell, getnames(ES))
end

struct S_Ctrl <: EmpiricalSummary end
function (::S_Ctrl)(θ::ParameterVector; n=500, y=zeros(Float64, 3, n))
    S = (
        TrajectoryRandomVariable(T_polarise(pbar2(θ)), P_Ctrl_0(θ)), 
        TrajectoryRandomVariable(T_depolarise(pbar2(θ)), P_Ctrl_1(θ)),
        TrajectoryRandomVariable(IsPolarised(pbar2(θ),long_tspan), P_Ctrl_0(θ)),
    )
    D = mean(y, S...)
end
Base.length(::S_Ctrl)=3
getnames(::Type{S_Ctrl}) = (:T_polarise, :T_depolarise, :IsPolarised)


struct S_Switch <: EmpiricalSummary end
function (::S_Switch)(θ::ParameterVector; n=500, y=zeros(Float64, 3, n))
    S = (
        TrajectoryRandomVariable(T_depolarise(pbar2(θ)), P_Switched(θ)),
        TrajectoryRandomVariable(T_neg(pbar2(θ)), P_Switched(θ)),
        TrajectoryRandomVariable(T1_lt_T2(T_perp(pbar2(θ)), T_neg(pbar2(θ))), P_Switched(θ)),
    )
    D = mean(y, S...)
end
Base.length(::S_Switch)=3
getnames(::Type{S_Switch}) = (:T_depolarise, :T_neg, :T1_lt_T2)

struct S_Stop <: EmpiricalSummary end
function (::S_Stop)(θ::ParameterVector; n=500, y=zeros(Float64, 3, n))
    S = (
        TrajectoryRandomVariable(T_depolarise(pbar2(θ)), P_NoEF_1(θ)),
        TrajectoryRandomVariable(T_neg(pbar2(θ)), P_NoEF_1(θ)),
        TrajectoryRandomVariable(T1_lt_T2(T_perp(pbar2(θ)), T_neg(pbar2(θ))), P_NoEF_1(θ)),
    )
    D = mean(y, S...)
end
Base.length(::S_Stop)=3
getnames(::Type{S_Stop}) = (:T_depolarise, :T_neg, :T1_lt_T2)


#const stat_list_switch = [
#    θ->TrajectoryRandomVariable(T_depolarise(pbar(θ)), P_Switched(θ)), 
#    θ->TrajectoryRandomVariable(T_neg(pbar(θ)), P_Switched(θ)), 
#    θ->TrajectoryRandomVariable(T1_lt_T2(T_perp(pbar(θ)), T_neg(pbar(θ))), P_Switched(θ)), 
#]
#const stat_list_stop = [
#    θ->TrajectoryRandomVariable(T_depolarise(pbar(θ)), P_NoEF_1(θ)), 
#    θ->TrajectoryRandomVariable(T_neg(pbar(θ)), P_NoEF_1(θ)), 
#    θ->TrajectoryRandomVariable(T1_lt_T2(T_perp(pbar(θ)), T_neg(pbar(θ))), P_NoEF_1(θ)), 
#]
