export log_partition_function

function log_partition_function(
    L::SyntheticLogLikelihood, 
    π::Prior{Names}, 
    q::Importance{Names},
    N::Int
) where Names

    P = Parameters(q, N)

    log_π = logpdf(π, P)
    log_q = logpdf(q, P)
    isalive = isfinite.(log_π)

    f(a, θ) = a ? L(θ) : -Inf
    log_sl = @showprogress pmap(f, isalive, ParameterSet(P))
    ell = log_sl .+ log_π .- log_q
    max_ell = maximum(ell)
    ell .-= max_ell
    W = mean(exp, ell)
    return log(W) + max_ell
end

function log_partition_function(
    L::SyntheticLogLikelihood,
    X,
    B::InferenceBatch,
    N::Int
)
    log_partition_function(L, Prior(X), Importance(B), N)
end