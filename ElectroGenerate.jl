module ElectroGenerate

include("ElectroInference.jl")
using .ElectroInference

############ Inference data

# Start with NoEF
function Posterior_NoEF(fn::String="electro_data")
    B = smc(L_NoEF(), Prior(), 5000, N_T=2000, alpha=0.8, Δt_min=1e-2)
    save(B, L_NoEF, fn)
    return B
end

# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function IntermediatePrior(X, fn::String="electro_data"; kwargs...)
    B = load(fn, L_NoEF, (:v, :EB_on, :EB_off, :D))
    return Sequential(B, X)
end
function SequentialPosterior_EF(
    X,
    fn::String="electro_data";
    kwargs...
)
    B = smc(L_EF(), IntermediatePrior(X, fn=fn), 5000, N_T=2000, alpha=0.8, Δt_min=1e-2)
    save(b, L_EF, fn)
    return B
end
function AllSequentialInference(fn::String="electro_data"; kwargs...)
    for X in combination_powerset
        SequentialPosterior_EF(X, fn; kwargs...)
    end
    return nothing
end

# Set up a joint inference problem, with the (known best, as default) parameter space X
function Posterior_Joint(
    X=[1,2,4],
    fn::String="electro_data";
    kwargs...
)
    B = smc(L_Joint(), Prior(X), 5000, N_T=2000, alpha=0.8, Δt_min=1e-2)
    save(B, L_Joint, fn)
    return B
end

end
