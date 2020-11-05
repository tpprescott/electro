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

export par_names, par_names_NoEF
export prior_support
export combination_powerset
export get_par_names

const par_names = (:v, :EB_on, :EB_off, :D, :γ1, :γ2, :γ3, :γ4)
const par_names_NoEF = par_names[1:4]
## Define possible priors
const prior_support = [
    Uniform(0,5),
    Uniform(0,5),
    Uniform(0,5),
    Uniform(0,0.5),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
]
const combination_powerset = powerset([1,2,3,4])
get_par_names(X) = par_names[[1,2,3,4,(Int(4).+X)...]]

include("observations.jl")
include("parameters.jl")
include("inputs.jl")
include("summaries.jl")
include("stochastic_simulations.jl")
include("synthetic_likelihoods.jl")
include("conditional_expectations.jl")
include("inference_batch.jl")
include("selection.jl")
include("io.jl")
include("recipes.jl")

export u_NoEF, u_EF, u_switch, u_stop
export P_NoEF, P_EF
export Y_NoEF, Y_EF
export xobs_NoEF, xobs_EF
export yobs_NoEF, yobs_EF

const u_NoEF = NoEF()
const u_EF = ConstantEF(1)
const u_switch = StepEF(1, -1, 90)
const u_stop = StepEF(1, 0, 90)

P_NoEF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), NoEF())
P_EF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), ConstantEF(1))

# Pixel size is 1.055125 μm for the NoEF data, and 0.91899 μm for (most of) the EF data

Y_NoEF(θ) = TrajectoryRandomVariable(InferenceSummary(1.055125), P_NoEF(θ))
Y_EF(θ) = TrajectoryRandomVariable(InferenceSummary(0.91899), P_EF(θ))

const xobs_NoEF = observation_filter(CSV.read("No_EF.csv"))
const xobs_EF = observation_filter(CSV.read("With_EF.csv"))
# Summarise - data is already pixellated, no need to do so again.
const yobs_NoEF = summarise(xobs_NoEF, InferenceSummary())
const yobs_EF = summarise(xobs_EF, InferenceSummary())

# For analysis purposes
# ConditionalExpectation(b_NoEF, S_NoEF(), n=500)

P_switch(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), u_switch)
P_stop(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), u_stop)

const long_tspan = (0.0, 1800.0)
P_NoEF_0(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(0), NoEF(), tspan=long_tspan)
P_NoEF_1(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), NoEF(), tspan=long_tspan)
P_Switched(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), ConstantEF(-1), tspan=long_tspan)


end
