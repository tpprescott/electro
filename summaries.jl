export AbstractSummary
export get_options
export summarise, summarise!
export InferenceSummary

export AnalysisSummary, HittingTime
export T_polarise, T_depolarise, T_neg, T_pos, T_perp
export IsPolarised
export T1_lt_T2

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
    δ::Float64
    function InferenceSummary(δ=eps())
        δ>0 || error("Needs positive pixel length")
        return new(δ)
    end
end
Base.length(::InferenceSummary)=4
Base.eltype(::InferenceSummary)=Float64

get_options(::InferenceSummary)=(saveat=5, save_idxs=2)
function (Y::InferenceSummary)(y, sol)
    Y(y, sol.u)
end

function (Y::InferenceSummary)(y, x::AbstractVector{ComplexF64})
    pixelated_x = round.(x./Y.δ).*Y.δ

    dx = diff(pixelated_x)
    td = sum(dx)
    y[1] = abs(td)
    y[2] = sum(abs, dx)
    y[3] = sqrt(sum(abs2, dx))
    y[4] = atan(imag(td), real(td))
    y
end

############################
# FOR analysis
############################

abstract type AnalysisSummary <: AbstractSummary end
abstract type HittingTime <: AnalysisSummary end

struct T_polarise <: HittingTime
    pbar2::Float64
end
struct T_depolarise <: HittingTime
    pbar2::Float64
end
struct T_perp <: HittingTime
    pbar2::Float64
end
struct T_neg <: HittingTime
    pbar2::Float64
end
struct T_pos <: HittingTime
    pbar2::Float64
end

Base.length(::AnalysisSummary)=1
Base.eltype(::AnalysisSummary)=Float64
get_options(::AnalysisSummary)=(save_idxs=1,)

ishit(Y::HittingTime, p::ComplexF64) = Y(p)
function (Y::HittingTime)(sol)
    h(p) = ishit(Y,p)
    t_out = count(h, sol.u)==0 ? sol.t[end] : getindex(sol.t, findfirst(h, sol.u))
    t_out
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

(Y::T_polarise)(p::ComplexF64) = abs2(p)>Y.pbar2
(Y::T_depolarise)(p::ComplexF64) = abs2(p)<=Y.pbar2
(Y::T_perp)(p::ComplexF64) = (abs2(p)>Y.pbar2)&&(abs(imag(p))>abs(real(p)))
(Y::T_pos)(p::ComplexF64) = (abs2(p)>Y.pbar2)&&(abs(imag(p))<real(p))
(Y::T_neg)(p::ComplexF64) = (abs2(p)>Y.pbar2)&&(-abs(imag(p))>real(p))

function _proportion(t, YY::NTuple{N, HittingTime}, sol::EnsembleSolution) where N
    tp = get_timepoint(sol, t)
    return [mean(Y, tp) for Y in YY]
end

###########

struct IsPolarised <: AnalysisSummary
    pbar2::Float64
    tspan::Tuple{Float64,Float64}
end
get_options(Π::IsPolarised) = (save_idxs=1, saveat=[Π.tspan[1], Π.tspan[2]])

(Π::IsPolarised)(sol) = abs2(sol[end])>Π.pbar2
function (Π::IsPolarised)(y, sol)
    y[1] = Π(sol)
end

####### 

struct T1_lt_T2{T1<:HittingTime, T2<:HittingTime} <: AnalysisSummary
    Time1::T1
    Time2::T2
end
function _make_callback(Y::T1_lt_T2)
    condition(u,t,integrator) = ishit(Y.Time1, u[1]) || ishit(Y.Time2, u[1])
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
end
get_options(Y::T1_lt_T2) = (save_idxs=1, callback=_make_callback(Y))

function (Π::T1_lt_T2)(sol)
    # Assumes (see callback) we stopped the simulation at one of the two hitting times
    p = sol[end]
    if Π.Time1(p)
        return true
    elseif Π.Time2(p)
        return false
    else
        @info "Neither region hit by t = $(sol.t[end])"
        return rand([true, false])
        # Should implement an error type that tells simulator to carry on
        # As with hitting times above...
    end
end
function (Π::T1_lt_T2)(y, sol) 
    y[1] = Π(sol)
end