module ElectroGenerate

include("ElectroInference.jl")
using .ElectroInference

############ Inference data

# Start with NoEF
function Posterior_NoEF(fn::String="electro_data")
    smc_NoEF = SMCProblem(L_NoEF(), Prior(), Tuple(100:50:550), Tuple(50:50:500), Tuple((2.0.^(1:10))/1024))
    b, I = sample(smc_NoEF, 100)
    save(b, L_NoEF, fn)
    return b, I
end

# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function IntermediatePrior(X, fn::String="electro_data")
    b = load(fn, L_NoEF, (:v, :EB_on, :EB_off, :D))
    I = isaccepted(b)
    π = Sequential(b[I], X)
    return π
end
function SequentialPosterior_EF(X, fn::String="electro_data")
    π_seq = IntermediatePrior(X, fn)
    smc_seq = SMCProblem(L_EF(), π, Tuple(100:50:550), Tuple(50:50:500), Tuple((2.0.^(1:10))/1024))
    b, I = sample(smc_seq, 100)
    save(b, L_EF, fn)
    return b, I
end
function AllSequentialInference(fn::String="electro_data")
    for X in combination_powerset
        SequentialPosterior_EF(X, fn)
    end
    return nothing
end

# Set up a joint inference problem, with the (known best, as default) parameter space X
function Posterior_Joint(X=[1,2,4], fn::String="electro_data")
    π = Prior(X)
    smc_Joint = SMCProblem(L_Joint(), π, Tuple(100:50:550), Tuple(50:50:500), Tuple((2.0.^(1:10))/1024))
    b, I = sample(smc_Joint, 100)
    save(b, L_Joint, fn)
end

end