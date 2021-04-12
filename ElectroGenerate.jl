include("ElectroInference.jl")
module ElectroGenerate

using ..ElectroInference
using Combinatorics

############ Inference data

# Start with NoEF
function Posterior_NoEF(; data, fn::String)
    B = smc(L_Ctrl(; data=data), Prior(), 5000, N_T=1000, alpha=0.6, Δt_min=1e-3, σ=[0.1, 0.05, 0.01])
    save(B, :L_Ctrl; fn=fn)
    return B
end

# Analyse NoEF results as a conditional expectation
function EmpiricalSummary_NoEF(; fn::String)
    B = load(:L_NoEF, (:v, :EB, :D); fn=fn)
    C = ConditionalExpectation(B, S_NoEF(), n=500)
    save(C, :S_NoEF; fn=fn)
    return C
end


# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function SequentialPosterior_EF(
    X;
    data,
    fn::String,
    kwargs...
)
    π_X = Prior(X)
    B0 = load(:L_NoEF, (:v, :EB, :D); fn=fn)

    σ = vcat([0.1, 0.05, 0.01], 0.05*ones(length(X)))
    B1 = InferenceBatch(π_X, B0, σ[1:3])
    smc(L_200(; data=data), π_X, 5000, B1, N_T=1000, alpha=0.8, Δt_min=1e-2, σ=σ)

    save(B1, :L_EF; fn=fn)
    return B1
end

function AllSequentialInference(; data, fn::String, kwargs...)
    for X in combination_powerset
        SequentialPosterior_EF(X; data=data, fn=fn, kwargs...)
    end
    return nothing
end
function AllSequentialPartitions(N::Int; data_NoEF, data_EF, fn::String, kwargs...)
    for X in combination_powerset
        Names = get_par_names(X)
        B = load(:L_EF, Names, fn=fn)
        big_ell = log_partition_function(L_Joint(data_NoEF=data_NoEF, data_EF=data_EF), X, B, N)
        @info "log_partition_function for $X is $big_ell"
        asave(big_ell, :log_partition_function, :L_EF, Names)
    end
end

# Analyse Switch results as a conditional expectation
function EmpiricalSummary_Switch(; fn::String="electro_data")
    B = load(:L_EF, get_par_names([1,2,4]); fn=fn)
    C = ConditionalExpectation(B, S_Switch(), n=500)
    save(C, :S_Switch; fn=fn)
    return C
end
# Analyse Stop results as a conditional expectation
function EmpiricalSummary_Stop(; fn::String="electro_data")
    B = load(:L_EF, get_par_names([1,2,4]); fn=fn)
    C = ConditionalExpectation(B, S_Stop(), n=500)
    save(C, :S_Stop; fn=fn)
    return C
end


# Set up a joint inference problem, with the (known best, as default) parameter space X
#function Posterior_Joint(
#    X=[1,2,4],
#    fn::String="electro_data";
#    kwargs...
#)
#    B = smc(L_Joint(), Prior(X), 2000, N_T=1000, alpha=0.8, Δt_min=1e-3)
#    save(B, :L_Joint, fn)
#    return B
#end

end
