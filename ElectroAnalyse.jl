include("ElectroInference.jl")
module ElectroAnalyse

using ..ElectroInference

# NoEF
const b_NoEF = load("electro_data", L_NoEF, (:v, :EB_on, :EB_off, :D))

end