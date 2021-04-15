include("ElectroInference.jl")
module ElectroGenerate

using ..ElectroInference
using Combinatorics

############ Inference data

# Start with NoEF
function Posterior_Ctrl(; kwargs...)
    SL_list = [LCtrl(data=yobs_Ctrl_1), LCtrl(data=yobs_Ctrl_2), LCtrl(data=yobs_Ctrl)]
    fn_list = ["replicate_1", "replicate_2", "merged_data"]
    p = Prior()
    for (L, fn) in zip(SL_list, fn_list)
        B = smc(L, p, 1000; synthetic_likelihood_n=500, N_T=333, alpha=0.8, Δt_min=1e-6, σ=[0.1, 0.05, 0.01], kwargs...)
        save(B, :L_Ctrl; fn=fn)
        mcmc!(B, 100, L, p, 500)
        save(B, :L_Ctrl; fn=fn*"_post")
    end
    println("Success! Control posteriors all done")
    return nothing
end

# Analyse NoEF results as a conditional expectation
function EmpiricalSummary_Ctrl()
    Φ = S_Ctrl()
    fn_list = ["replicate_1", "replicate_2", "merged_data"]
    for fn in fn_list
        B = load(:L_Ctrl, (:v, :EB, :D); fn=fn*"_post")
        C = ConditionalExpectation(B, S_NoEF(); n=500)
        save(C, :S_Ctrl; fn=fn*"_post")
    end
    println("Success! Control conditional expectations all done")
    return nothing
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
