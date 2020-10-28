module ElectroGenerate

include("ElectroInference.jl")
using .ElectroInference

############ Inference data

const b_s = Tuple(fill(100,10))
const n_s_l = Tuple(fill(500,10))
const t = Tuple((2.0.^(1:10))./1024)
const sample_size = 100

# Start with NoEF
function Posterior_NoEF(
    fn::String="electro_data";
    batch_sizes=b_s,
    n_synthetic_likelihoods=n_s_l,
    temperatures=t,
    kwargs...
)
    smc_NoEF = SMCProblem(L_NoEF(), Prior(), batch_sizes, n_synthetic_likelihoods, temperatures)
    b, I = sample(smc_NoEF, sample_size)
    save(b, L_NoEF, fn)
    return b, I
end

# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function IntermediatePrior(X, fn::String="electro_data"; kwargs...)
    b = load(fn, L_NoEF, (:v, :EB_on, :EB_off, :D))
    I = isaccepted(b)
    π = Sequential(b[I], X)
    return π
end
function SequentialPosterior_EF(
    X,
    fn::String="electro_data";
    batch_sizes=b_s,
    n_synthetic_likelihoods=n_s_l,
    temperatures=t,
    kwargs...
)
    π_seq = IntermediatePrior(X, fn)
    smc_seq = SMCProblem(L_EF(), π, batch_sizes, n_synthetic_likelihoods, temperatures)
    b, I = sample(smc_seq, sample_size)
    save(b, L_EF, fn)
    return b, I
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
    batch_sizes=b_s,
    n_synthetic_likelihoods=n_s_l,
    temperatures=t,
    kwargs...
)
    π = Prior(X)
    smc_Joint = SMCProblem(L_Joint(), π, batch_sizes, n_synthetic_likelihoods, temperatures)
    b, I = sample(smc_Joint, sample_size)
    save(b, L_Joint, fn)
end

end