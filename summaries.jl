export AbstractSummary
export get_options
export summarise, summarise!
export InferenceSummary

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