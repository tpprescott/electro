function SciMLBase.__solve(prob::SciMLBase.AbstractEnsembleProblem,
                 alg::Union{SciMLBase.DEAlgorithm,Nothing},
                 ensemblealg::SciMLBase.BasicEnsembleAlgorithm;
                 trajectories, batch_size = trajectories,
                 pmap_batch_size = batch_size÷100 > 0 ? batch_size÷100 : 1, kwargs...)

  num_batches = trajectories ÷ batch_size
  num_batches < 1 && error("trajectories ÷ batch_size cannot be less than 1, got $num_batches")
  num_batches * batch_size != trajectories && (num_batches += 1)

  if num_batches == 1 && prob.reduction === SciMLBase.DEFAULT_REDUCTION
    elapsed_time = @elapsed u = SciMLBase.solve_batch(prob,alg,ensemblealg,1:trajectories,pmap_batch_size;kwargs...)
    _u = SciMLBase.tighten_container_eltype(u)
    return SciMLBase.EnsembleSolution(_u,elapsed_time,true)
  end

  converged::Bool = false
  i = 1
  II = (batch_size*(i-1)+1):batch_size*i

  batch_data = SciMLBase.solve_batch(prob,alg,ensemblealg,II,pmap_batch_size;kwargs...)

  u = prob.u_init === nothing ? similar(batch_data, 0) : prob.u_init
  u,converged = prob.reduction(u,batch_data,II)
  elapsed_time = @elapsed @progress for i in 2:num_batches
    converged && break
    if i == num_batches
      II = (batch_size*(i-1)+1):trajectories
    else
      II = (batch_size*(i-1)+1):batch_size*i
    end
    batch_data = SciMLBase.solve_batch(prob,alg,ensemblealg,II,pmap_batch_size;kwargs...)
    u,converged = prob.reduction(u,batch_data,II)
  end

  _u = SciMLBase.tighten_container_eltype(u)

  return SciMLBase.EnsembleSolution(_u,elapsed_time,converged)

end
