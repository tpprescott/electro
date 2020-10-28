export AbstractSummary
export get_options
export summarise, summarise!
export InferenceSummary

export AnalysisSummary, HittingTime
export T_polarise, T_depolarise, T_neg, T_pos, T_perp
export IsPolarised

abstract type AbstractSummary end
function Base.length(::AbstractSummary) 
    error("Implement 'Base.length' for the summary statistic!")
end
function Base.eltype(::AbstractSummary) 
    error("Implement 'Base.eltype' for the summary statistic!")
end
get_options(Y::AbstractSummary) = NamedTuple()


function summarise(x::AbstractMatrix, Y::AbstractSummary)
    y = zeros(eltype(Y), length(Y), size(x,2))
    summarise!(y, x, Y)
end
function summarise(x::EnsembleSolution, Y::AbstractSummary)
    y = zeros(eltype(Y), length(Y), length(x))
    summarise!(y, x, Y)
end
function summarise(x::AbstractVector, Y::AbstractSummary)
    y = zeros(eltype(Y), length(Y))
    summarise!(y, x, Y)
end
function summarise(x::DESolution, Y::AbstractSummary)
    y = zeros(eltype(Y), length(Y))
    summarise!(y, x, Y)
end

function summarise!(y::AbstractMatrix, x::AbstractMatrix, Y::AbstractSummary)
    map(Y, eachcol(y), eachcol(x))
    y
end
function summarise!(y::AbstractMatrix, x::EnsembleSolution, Y::AbstractSummary)
    map(Y, eachcol(y), x)
    y
end
function summarise!(y::AbstractVector, x::AbstractVector, Y::AbstractSummary)
    Y(y,x)
end
function summarise!(y::AbstractVector, x::DESolution, Y::AbstractSummary)
    Y(y,x)
end


############################
# FOR inference specifically
############################

struct InferenceSummary <: AbstractSummary
end
Base.length(::InferenceSummary)=4
Base.eltype(::InferenceSummary)=Float64

get_options(::InferenceSummary)=(saveat=5, save_idxs=2)
function (Y::InferenceSummary)(y, sol)
    Y(y, sol.u)
end

function (Y::InferenceSummary)(y, x::AbstractVector{ComplexF64})
    dx = diff(x)
    td = sum(dx)
    y[1] = abs(td)
    y[2] = mean(abs, dx)
    y[3] = std(broadcast(abs, dx))
    y[4] = atan(imag(td), real(td))
    y
end

############################
# FOR analysis
############################

abstract type AnalysisSummary <: AbstractSummary end
abstract type HittingTime <: AnalysisSummary end

struct T_polarise <: HittingTime
    pbar::Float64
end
struct T_depolarise <: HittingTime
    pbar::Float64
end
struct T_perp <: HittingTime
    pbar::Float64
end
struct T_neg <: HittingTime
    pbar::Float64
end
struct T_pos <: HittingTime
    pbar::Float64
end

Base.length(::AnalysisSummary)=1
Base.eltype(::AnalysisSummary)=Float64
get_options(::AnalysisSummary)=(save_idxs=1,)

ishit(Y::HittingTime, p::ComplexF64) = Y(p)
function (Y::HittingTime)(sol)
    getindex(sol.t, findfirst(p->ishit(Y,p), sol.u))
end
function (Y::HittingTime)(y, sol)
    y[1] = Y(sol)
end

function _make_callback(Y::HittingTime)
    condition(u,t,integrator) = ishit(Y, u[1])
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
end
get_options(Y::HittingTime) = (save_idxs=1, callback=_make_callback(Y))

(Y::T_polarise)(p::ComplexF64) = abs(p)>Y.pbar
(Y::T_depolarise)(p::ComplexF64) = abs(p)<=Y.pbar
(Y::T_perp)(p::ComplexF64) = (abs(p)>Y.pbar)&&(abs(imag(p))>abs(real(p)))
(Y::T_pos)(p::ComplexF64) = (abs(p)>Y.pbar)&&(abs(imag(p))<real(p))
(Y::T_neg)(p::ComplexF64) = (abs(p)>Y.pbar)&&(-abs(imag(p))>real(p))

###########

struct IsPolarised <: AnalysisSummary
    pbar::Float64
    t_inf::Float64
end

(Π::IsPolarised)(sol) = abs(sol(Π.t_inf))>Π.pbar
function (Π::IsPolarised)(y, sol)
    y[1] = Π(sol)
end