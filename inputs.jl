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

struct NStepEF <: AbstractEMField
    x::Vector{ComplexF64}
    t::Vector{Float64}
    function NStepEF(x, t; x0=complex(0.0))
        length(x)==length(t) || @error "Need x and t to be equal lengths; supply x0 as kwarg if needed."
        sort!(t)
        return new(vcat(x0, x), vcat(0.0, t))
    end
end
function (emf::NStepEF)(t)
    j = searchsortedlast(emf.t, t)
    return iszero(j) ? error("But t<0!") : emf.x[j]
end