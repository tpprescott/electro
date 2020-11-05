include("ElectroInference.jl")
module ElectroGenerate

using ..ElectroInference
using Combinatorics

############ Inference data

# Start with NoEF
function Posterior_NoEF(; fn::String="electro_data")
    B = smc(L_NoEF(), Prior(), 2000, N_T=1000, alpha=0.8, Δt_min=1e-2)
    save(B, :L_NoEF; fn=fn)
    return B
end

# Analyse NoEF results as a conditional expectation
function EmpiricalSummary_NoEF(; fn::String="electro_data")
    B = load(:L_NoEF, (:v, :EB_on, :EB_off, :D); fn=fn)
    C = ConditionalExpectation(B, S_NoEF(), n=500)
    save(C, :S_NoEF; fn=fn)
    return C
end


# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function SequentialPosterior_EF(
    X;
    fn::String="electro_data",
    kwargs...
)
    π_X = Prior(X)
    B0 = load(:L_NoEF, (:v, :EB_on, :EB_off, :D); fn=fn)

    B1 = InferenceBatch(π_X, B0)
    smc(L_EF(), π_X, 2000, B1, N_T=1000, alpha=0.8, Δt_min=1e-2)

    save(B1, :L_EF; fn=fn)
    return B1
end

function AllSequentialInference(; fn::String="electro_data", kwargs...)
    for X in combination_powerset
        SequentialPosterior_EF(X; fn=fn, kwargs...)
    end
    return nothing
end
function AllSequentialPartitions(N::Int; fn::String="electro_data", kwargs...)
    for X in combination_powerset
        Names = get_par_names(X)
        B = load(:L_EF, Names, fn=fn)
        big_ell = log_partition_function(L_Joint(), X, B, N)
        @info "log_partition_function for $X is $big_ell"
        asave(big_ell, :log_partition_function, :L_EF, Names)
    end
end

# Set up a joint inference problem, with the (known best, as default) parameter space X
function Posterior_Joint(
    X=[1,2,4],
    fn::String="electro_data";
    kwargs...
)
    B = smc(L_Joint(), Prior(X), 2000, N_T=1000, alpha=0.8, Δt_min=1e-3)
    save(B, :L_Joint, fn)
    return B
end

end
