include("ElectroInference.jl")
module ElectroGenerate

using ..ElectroInference
using Combinatorics, Distributions, Statistics

############ Inference data

# Start with NoEF
function Posterior_Ctrl(; kwargs...)
    SL_list = (L_Ctrl(data=yobs_Ctrl_1), L_Ctrl(data=yobs_Ctrl_2), L_Ctrl(data=yobs_Ctrl))
    fn_list = ("replicate_1", "replicate_2", "merged_data")
    p = Prior()
    for (L, fn) in zip(SL_list, fn_list)
        B = smc(L, p, 1000; synthetic_likelihood_n=500, N_T=333, alpha=0.8, Δt_min=1e-6, σ=[0.1, 0.05, 0.01], kwargs...)
        save(B, :L_Ctrl; fn=fn)
        mcmc!(B, 100, L, p, 500)
        save(B, :L_Ctrl; fn=fn*"_post")
    end
    println("Success! Control posteriors all done")
    return true
end

# Analyse NoEF results as a conditional expectation
function EmpiricalSummary_Ctrl()
    Φ = S_Ctrl()
    fn_list = ("replicate_1", "replicate_2", "merged_data")
    for fn in fn_list
        B = load(:L_Ctrl, (:v, :EB, :D); fn=fn*"_post")
        C = ConditionalExpectation(B, S_Ctrl(); n=500)
        save(C, :S_Ctrl; fn=fn*"_post")
    end
    println("Success! Control conditional expectations all done")
    return true
end


# Set up the intermediate prior based on the NoEF output and evaluated against EF only
function Posterior_EF(X; kwargs...)
    p = Prior(X)
    σ=[0.1, 0.05, 0.01]
    for i in X
        push!(σ, 0.1)
    end

    dataCtrl_list = (yobs_Ctrl_1, yobs_Ctrl_2, yobs_Ctrl)
    data200_list = (yobs_200_1, yobs_200_2, yobs_200)
    
    SL_list = (L_Joint(data_NoEF=a, data_EF=b) for (a,b) in zip(dataCtrl_list, data200_list))
    fn_list = ("replicate_1", "replicate_2", "merged_data")
    
    for (L, fn) in zip(SL_list, fn_list)
        B = smc(L, p, 1000; synthetic_likelihood_n=500, N_T=333, alpha=0.8, Δt_min=1e-6, σ=σ, kwargs...)
        save(B, :L_Joint; fn=fn)
        mcmc!(B, 100, L, p, 500)
        save(B, :L_Joint; fn=fn*"_post")
    end
    println("Success! Posteriors for $X all done")
    return true
end

function AllPosterior_EF(; kwargs...)
    for X in powerset([1,2,3,4])
        Posterior_EF(X; kwargs...)
    end
    println("Success! All posteriors are done!")
    return true
end

function PosteriorPartition_EF(L, X, N; synthetic_likelihood_n=500, kwargs...)
    Names = get_par_names(X)
    B = load(:L_Joint, Names, fn="merged_data_post")
    prior = Prior(X)

    # Form and sample from importance distribution
    Σ = cov(B.θ)
    q = MixtureModel(map(p->MvNormal(p.θ.θ, Σ), B))
    ts = rand(q, N)
    θs = Parameters.(eachcol(ts), Ref(Names))

    # Get importance weights
    logqs = logpdf(q, ts)
    logps = logpdf(prior, ts)
    simFlags = insupport.(Ref(prior), θs)

    # Simulate and get importance weighted posterior ∫ L(θ) π(θ) dθ = ∫ L(θ) [π(θ)/q(θ)] q(θ) dθ
    logsl = get_log_sl(L, θs, simFlags, synthetic_likelihood_n)
    ell = logsl .+ logps .- logqs
    ellmax = maximum(ell)
    big_ell = log(mean(exp, ell.-ellmax)) + ellmax

    asave(big_ell, :log_partition_function, :L_Joint, Names, fn="merged_data_post")
    @info "Success! log_partition_function for $X is $big_ell"
end
function AllPosteriorPartitions(N::Int; synthetic_likelihood_n=500, kwargs...)
    L = L_Joint(; data_NoEF=yobs_Ctrl, data_EF=yobs_200)
    for X in powerset([1,2,3,4])
        PosteriorPartition_EF(L, X, N, synthetic_likelihood_n=synthetic_likelihood_n, kwargs...)
    end
    println("Success! All partition functions calculated!")
    return true
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
