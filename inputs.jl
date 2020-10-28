export AbstractEMField
export NoEF, ConstantEF, StepEF

abstract type AbstractEMField end

struct NoEF <: AbstractEMField end
(::NoEF)(t) = complex(0.0)

struct ConstantEF <: AbstractEMField
    x::Complex{Float64}
end
(emf::ConstantEF)(t) = emf.x

struct StepEF <: AbstractEMField
    x0::Complex{Float64}
    x1::Complex{Float64}
    t_step::Float64
end
(emf::StepEF)(t) = t<emf.t_step ? emf.x0 : emf.x1

