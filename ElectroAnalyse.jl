module ElectroAnalyse

include("ElectroInference.jl")
using .ElectroInference

# NoEF
const b_NoEF = load("electro_data", L_NoEF, (:v, :EB_on, :EB_off, :D))

struct FixedInitialPolarity <: AbstractPolarityDistribution
    p0::ComplexF64
end
(r::FixedInitialPolarity)() = [r.p0, zero(ComplexF64)]

const long_T = 1800.0
const tspan = (0.0, long_T)
P_NoEF_0(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(0), NoEF(), tspan=tspan)
P_NoEF_1(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), NoEF(), tspan=tspan)
P_Switched(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), ConstantEF(-1), tspan=tspan)

pbar2(theta) = ElectroInference.get_pbar2(theta[2], theta[3])

struct S_NoEF <: EmpiricalSummary end
function (::S_NoEF)(θ::ParameterVector; n=500, y=zeros(Float64, 3, n))
    S = (
        TrajectoryRandomVariable(T_polarise(pbar2(θ)), P_NoEF_0(θ)), 
        TrajectoryRandomVariable(T_depolarise(pbar2(θ)), P_NoEF_1(θ)),
        TrajectoryRandomVariable(IsPolarised(pbar2(θ),tspan), P_NoEF_0(θ)),
    )
    D = mean(y, S...)
end
Base.length(::S_NoEF)=3

function EmpiricalSummary_NoEF()
    return ConditionalExpectation(b_NoEF, S_NoEF(), n=500)
end

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


end