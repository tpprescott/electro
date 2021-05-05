@everywhere include("simplifier.jl")
using .SimpleElectro
using FileIO, JLD2

function generateSample(fun, vn, fn)
    θ = fun()
    save(string(fn), string(vn), θ)
    @info "$vn done!"
    return nothing
end

generateSample(tup) = generateSample(tup..., "simpleOutput.jld2")
fun_list_Ctrl = [
    (θ_Ctrl, :Ctrl),
    (θ_Ctrl_1, :Ctrl_1),
    (θ_Ctrl_2, :Ctrl_2),
]

fun_list_Joint = vcat(
    [(θ_Joint[X], Symbol(:Joint_g, X...)) for X in Xs],
    [(θ_Joint_1[X], Symbol(:Joint_1_g, X...)) for X in Xs],
    [(θ_Joint_2[X], Symbol(:Joint_2_g, X...)) for X in Xs],
)

generateSample(fun_list_Ctrl[1])
generateSample(fun_list_Joint[5])
map(generateSample, fun_list_Ctrl[2:3])
map(generateSample, fun_list_Joint[1:4])
map(generateSample, fun_list_Joint[6:end])
