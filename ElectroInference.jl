module ElectroInference

using Distributions, Random, Combinatorics, InvertedIndices
using Statistics, StatsBase
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using DataFrames, CSV
using Distributed, ProgressMeter
using Polynomials, Roots, LinearAlgebra
using HDF5
using RecipesBase, LaTeXStrings

export rand, pdf, logpdf, sample, mean

export par_names, par_names_NoEF
export prior_support
export combination_powerset
export get_par_names

const par_names = (:v, :EB, :D, :γ1, :γ2, :γ3, :γ4)
const par_names_NoEF = par_names[1:3]
## Define possible priors
const prior_support = [
    Uniform(0,5),
    Uniform(0,5),
    Uniform(0,0.5),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
    Uniform(0,2),
]
const combination_powerset = powerset([1,2,3,4])
get_par_names(X) = par_names[[1,2,3,(Int(3).+X)...]]

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
export P_Ctrl, P_200#, P_NoEF, P_EF
export Y_Ctrl, Y_200#, Y_NoEF, Y_EF
# export xobs_NoEF, xobs_EF
# export yobs_NoEF, yobs_EF
export xobs_Ctrl, xobs_Ctrl_1, xobs_Ctrl_2
export yobs_Ctrl, yobs_Ctrl_1, yobs_Ctrl_2
export xobs_200, xobs_200_1, xobs_200_2
export yobs_200, yobs_200_1, yobs_200_2

const u_NoEF = NoEF()
const u_EF = ConstantEF(1)
const u_switch = StepEF(1, -1, 90)
const u_stop = StepEF(1, 0, 90)

# P_NoEF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), NoEF())
# P_EF(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), ConstantEF(1))
P_Ctrl(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), NoEF(), tspan=(0.0,300.0))
P_200(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), StepEF(0, 1, 60), tspan=(0.0,180.0))

# Pixel size is 1.055125 μm for the NoEF data, and 0.91899 μm for (most of) the EF data

# Y_NoEF(θ) = TrajectoryRandomVariable(InferenceSummary(1.055125), P_NoEF(θ))
# Y_EF(θ) = TrajectoryRandomVariable(InferenceSummary(0.91899), P_EF(θ))
Y_Ctrl(θ) = TrajectoryRandomVariable(InferenceSummary(), P_Ctrl(θ))
Y_200(θ) = TrajectoryRandomVariable(IntervalsSummary([1:13, 13:37]), P_200(θ))

# const xobs_NoEF = observation_filter(CSV.File("No_EF.csv"))
# const xobs_EF = observation_filter(CSV.File("With_EF.csv"))
const xobs_Ctrl_1 = observation_filter(CSV.File("data/test/Ctrl-1.csv"), ninterval=61)
const xobs_Ctrl_2 = observation_filter(CSV.File("data/test/Ctrl-2.csv"), ninterval=61)
const xobs_Ctrl = hcat(xobs_Ctrl_1, xobs_Ctrl_2)
const xobs_200_1 = observation_filter(CSV.File("data/test/200mV-1.csv"), ninterval=61)
const xobs_200_2 = observation_filter(CSV.File("data/test/200mV-2.csv"), ninterval=61)
const xobs_200 = hcat(xobs_200_1, xobs_200_2)

# Summarise - data is already pixellated, no need to do so again.
# const yobs_NoEF = summarise(xobs_NoEF, InferenceSummary())
# const yobs_EF = summarise(xobs_EF, InferenceSummary())
const yobs_Ctrl_1 = summarise(xobs_Ctrl_1, InferenceSummary())
const yobs_Ctrl_2 = summarise(xobs_Ctrl_2, InferenceSummary())
const yobs_Ctrl = hcat(yobs_Ctrl_1, yobs_Ctrl_2)
const yobs_200_1 = summarise(xobs_200_1, IntervalsSummary([1:13, 13:37]))
const yobs_200_2 = summarise(xobs_200_2, IntervalsSummary([1:13, 13:37]))
const yobs_200 = hcat(yobs_200_1, yobs_200_2)

# For analysis purposes
# ConditionalExpectation(b_NoEF, S_NoEF(), n=500)
export P_switch, P_stop
export P_Ctrl_0, P_Ctrl_1, P_Switched

P_switch(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), u_switch)
P_stop(θ) = TrajectoryDistribution(θ, RandomInitialPolarity(0.1), u_stop)

const long_tspan = (0.0, 1800.0)
P_Ctrl_0(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(0), NoEF(), tspan=long_tspan)
P_Ctrl_1(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), NoEF(), tspan=long_tspan)
P_Switched(θ) = TrajectoryDistribution(θ, FixedInitialPolarity(1), ConstantEF(-1), tspan=long_tspan)


end
