include("ElectroInference.jl")
module ElectroAnalyse

using ..ElectroInference

function see_velocity(θ::NamedTuple{par_names})
    β, pbar2 = ElectroInference._map_barriers_to_coefficients(θ.EB_on, θ.EB_off)
    function polarity_density(p)
        exp(θ.γ4*real(p) - ElectroInference.W_poly(β, pbar2)(abs2(p)))
    end
    f! = ElectroInference.SDEdrift(ConstantEF(1); θ...)
end

# NoEF
const b_NoEF = load("electro_data", L_NoEF, (:v, :EB_on, :EB_off, :D))

end