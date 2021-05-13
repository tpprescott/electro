# Quantifying the impact of electric fields on single-cell motility
TP Prescott, K Zhu, M Zhao, and RE Baker (2021)

Code associated with the bioRxiv preprint:
https://www.biorxiv.org/content/10.1101/2021.01.22.427762v1

### Paper abstract
Cell motility in response to environmental cues forms the basis of many developmental processes in multicellular organisms. 
One such environmental cue is an electric field (EF), which induces a form of motility known as electrotaxis.
Electrotaxis has evolved in a number of cell types to guide wound healing, and has been associated with different cellular processes, suggesting that observed electrotactic behaviour is likely a combination of multiple distinct effects arising from the presence of an EF.
In order to determine the different mechanisms by which observed electrotactic behaviour emerges, and thus to design EFs that can be applied to direct and control electrotaxis, researchers require accurate quantitative predictions of cellular responses to externally-applied fields.
Here, we use mathematical modelling to formulate and parametrise a variety of hypothetical descriptions of how cell motility may change in response to an EF.
We calibrate our model to observed data using synthetic likelihoods and Bayesian sequential learning techniques, and demonstrate that EFs impact cellular motility in three distinct ways.
We also demonstrate how the model allows us to make predictions about cellular motility under different EFs.
The resulting model and calibration methodology will thus form the basis for future data-driven and model-based feedback control strategies based on electric actuation.

## Data
The experimental data sets for the autonomous and electrotactic experiments are `No_EF.csv` and `With_EF.csv`, respectively.
The output of the large-scale simulation runs used to generate the figures in the manuscript are in the HDF5 formatted data set `electro_data.h5`.

## Code
Figures are generated by the functions defined in `MakeFigs.jl`.

### Modules

`ElectroInference`
Defines the functions that do the sampling and simulation-based inference of the autonomous and electrotactic models.
Includes the functions defined in all other `*.jl` files.

`ElectroGenerate`
Additional functions that produce the data sets depicted in the paper.

`ElectroAnalyse`
Additional functions that load previously-generated data and define the types of figures used in the paper.

### Functions, types, files

`sde.jl` defines the specific model to be simulated through `DifferentialEquations.solve`, implemented as a `DifferentialEquations.EnsembleProblem` in `stochastic_simulations.jl`.  

`synthetic_likelihoods.jl` contains `empirical_fit` which maps the ensemble simulation of summary statistics to a `Distributions.MvNormal` (i.e. multivariate Gaussian) distribution. 
Then `SyntheticLogLikelihood(::ParameterVector)` runs by (a) simulating the model for that parameter a number of times (b) fitting the multivariate normal (c) getting the log-likelihood of the data point using that fitted distribution.

The function `smc` defined in `inference_batch.jl` can be used to implement Algorithm 1 from the Supplementary Information. The full two-part algorithm is implemented in full by a combination of `ElectroGenerate.Posterior_NoEF` followed by `ElectroGenerate.SequentialPosterior_EF`.
Note that the latter loads a previously-completed first part of the algorithm, generated by the former.