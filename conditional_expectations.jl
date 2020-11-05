# struct TrajectoryRandomVariable{YY<:AbstractSummary, PP<:TrajectoryDistribution} <: Sampleable{Multivariate, Continuous}
#     Y::YY # Summary statistic definition
#     P::PP # Trajectory distribution
# end
# struct TrajectoryDistribution{B<:SDEProblem, E<:EnsembleProblem}
#     base::B
#     prob::E
# end

export EmpiricalSummary, getnames
export S_NoEF
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
    w = Weights(c.ell .- maximum(c.ell))
    NamedTuple{Names}(vec(mean(c.D, w, dims=2)))
end

##### For analysis

pbar2(theta) = get_pbar2(theta[2], theta[3])

abstract type EmpiricalSummary end
function getnames(::Type{T}) where T<:EmpiricalSummary
    error("Define the names of the summary statistic.")
end

function ConditionalExpectation(θ::ParameterSet, ell, Φ::ES; n=500) where ES<:EmpiricalSummary
    ϕ(θ_i) = Φ(θ_i; n=n)
    Dvec = @showprogress pmap(ϕ, θ)
    D = hcat(Dvec...)
    return ConditionalExpectation(D, ell, getnames(ES))
end

struct S_NoEF <: EmpiricalSummary end
function (::S_NoEF)(θ::ParameterVector; n=500, y=zeros(Float64, 3, n))
    S = (
        TrajectoryRandomVariable(T_polarise(pbar2(θ)), P_NoEF_0(θ)), 
        TrajectoryRandomVariable(T_depolarise(pbar2(θ)), P_NoEF_1(θ)),
        TrajectoryRandomVariable(IsPolarised(pbar2(θ),long_tspan), P_NoEF_0(θ)),
    )
    D = mean(y, S...)
end
Base.length(::S_NoEF)=3
getnames(::Type{S_NoEF}) = (:T_polarise, :T_depolarise, :IsPolarised)

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
