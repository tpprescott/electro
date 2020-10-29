export InferenceProblem, SMCProblem

struct InferenceProblem{L<:SyntheticLogLikelihood, Π<:ParameterDistribution, Q<:ParameterDistribution}
    synthetic_likelihood::L
    prior::Π
    importance::Q
    batch_size::Int
    synthetic_likelihood_n::Int
    temperature::Real

    function InferenceProblem(sl::L, π::Π, q::Q=π; batch_size::Int, synthetic_likelihood_n::Int=500, temp=1.0) where L where Π<:ParameterDistribution{Names} where Q<:ParameterDistribution{Names} where Names
        return new{L, Π, Q}(sl, π, q, batch_size, synthetic_likelihood_n, temp)
    end
end

function Distributions.sample(prob::InferenceProblem, n::Int; max_batch::Int=n*10000)
    args = (
        prob.synthetic_likelihood,
        prob.prior,
        prob.importance,
    )
    
    n_inc = prob.batch_size
    b = InferenceBatch(args..., n_inc; synthetic_likelihood_n=prob.synthetic_likelihood_n)
    A_next = count(isaccepted(b, temp=prob.temperature))

    while (length(b) < max_batch)
        L = length(b)
        A = A_next
        @info "$A accepted from $L proposals;"
        InferenceBatch(b, args..., n_inc; synthetic_likelihood_n=prob.synthetic_likelihood_n)
        A_next = count(isaccepted(b, temp=prob.temperature))
        if A_next <= A
            n_inc *= 2
        elseif A_next >= n
            break
        end
    end
    I = isaccepted(b, temp=prob.temperature)
    return b, I
end

struct SMCProblem{N, L<:SyntheticLogLikelihood, Π<:ParameterDistribution}
    synthetic_likelihood::L
    prior::Π
    batch_size::NTuple{N,Int}
    synthetic_likelihood_n::NTuple{N, Int}
    temperature::NTuple{N,Real}
end

function Distributions.sample(prob::SMCProblem{N}, n::NTuple{N, Int}; max_batch::NTuple{N, Int}=n.*10000) where N
    q_t = prob.prior
    for t in 1:N
        prob_t = InferenceProblem(
            prob.synthetic_likelihood,
            prob.prior,
            q_t;
            batch_size=prob.batch_size[t],
            synthetic_likelihood_n=prob.synthetic_likelihood_n[t],
            temp=prob.temperature[t],
        )
        b, I = sample(prob_t, n[t]; max_batch=max_batch[t])

        if t<N
            q_t = Importance(b[I])
        else
            return b, I
        end
    end
end
Distributions.sample(prob::SMCProblem{N}, n::Int; max_batch::Int=n*10000) where N = sample(prob, Tuple(fill(n,N)), max_batch=Tuple(fill(max_batch,N)))