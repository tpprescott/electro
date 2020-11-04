include("ElectroInference.jl")
module ElectroGenerate

using ..ElectroInference
using Combinatorics

############ Inference data

# Start with NoEF
function Posterior_NoEF(fn::String="electro_data")
    B = smc(L_NoEF(), Prior(), 2000, N_T=1000, alpha=0.8, Δt_min=1e-2)
    save(B, L_NoEF, fn)
    return B
end

# Analyse NoEF results as a conditional expectation
function EmpiricalSummary_NoEF(fn::String="electro_data")
    B = load(fn, L_NoEF, (:v, :EB_on, :EB_off, :D))
    C = ConditionalExpectation(B, S_NoEF(), n=500)
    save(C, S_NoEF, fn)
    return C
end


# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function SequentialPosterior_EF(
    X,
    fn::String="electro_data";
    kwargs...
)
    B0 = load(fn, L_NoEF, (:v, :EB_on, :EB_off, :D))
    B1 = InferenceBatch(Prior(X), B0)
    smc(L_EF(), IntermediatePrior(X, fn=fn), 2000, B1, N_T=1000, alpha=0.8, Δt_min=1e-2)
    save(B1, L_EF, fn)
    return B1
end

const combination_powerset = powerset([1,2,3,4])
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
    B = smc(L_Joint(), Prior(X), 2000, N_T=1000, alpha=0.8, Δt_min=1e-2)
    save(B, L_Joint, fn)
    return B
end

end
