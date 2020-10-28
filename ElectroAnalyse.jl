module ElectroAnalyse

include("ElectroInference.jl")
using .ElectroInference

const b_NoEF = load("electro_data", L_NoEF, (:v, :EB_on, :EB_off, :D))

struct FixedInitialPolarity <: AbstractPolarityDistribution
    p0::ComplexF64
end
(r::FixedInitialPolarity)() = [p0, zero(ComplexF64)]

P_NoEF_0(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(0), NoEF())
P_NoEF_1(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), NoEF())
P_Switched(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), ConstantEF(-1))

end