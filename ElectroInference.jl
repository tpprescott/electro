module ElectroInference

using Distributions, Random, Combinatorics, InvertedIndices
using Statistics, StatsBase
using DifferentialEquations
using DataFrames, CSV
using Distributed, ProgressMeter
using Roots, LinearAlgebra
using HDF5
using RecipesBase, LaTeXStrings

export rand, pdf, logpdf, sample, mean

include("observations.jl")
include("parameters.jl")
include("inputs.jl")
include("summaries.jl")
include("stochastic_simulations.jl")
include("synthetic_likelihoods.jl")
include("conditional_expectations.jl")
include("inference_batch.jl")
include("io.jl")
include("recipes.jl")

export par_names, par_names_NoEF
export prior_support
export u_NoEF, u_EF, u_switch, u_stop
export P_NoEF, P_EF
export Y_NoEF, Y_EF
export xobs_NoEF, xobs_EF
export yobs_NoEF, yobs_EF
export combination_powerset

const par_names = (:v, :EB_on, :EB_off, :D, :γ1, :γ2, :γ3, :γ4)
const par_names_NoEF = par_names[1:4]

## Define possible priors
const prior_support = [
    Uniform(0,3),
    Uniform(0,5),
    Uniform(0,5),
    Uniform(0,0.5),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
]

const u_NoEF = NoEF()
const u_EF = ConstantEF(1)
const u_switch = StepEF(1, -1, 90)
const u_stop = StepEF(1, 0, 90)

P_NoEF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), NoEF())
P_EF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), ConstantEF(1))

const Y = InferenceSummary()
Y_NoEF(θ) = TrajectoryRandomVariable(Y, P_NoEF(θ))
Y_EF(θ) = TrajectoryRandomVariable(Y, P_EF(θ))

const xobs_NoEF = observation_filter(CSV.read("No_EF.csv"))
const xobs_EF = observation_filter(CSV.read("With_EF.csv"))
const yobs_NoEF = summarise(xobs_NoEF, Y)
const yobs_EF = summarise(xobs_EF, Y)

const combination_powerset = powerset([1,2,3,4])

end
