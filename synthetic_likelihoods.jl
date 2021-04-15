export empirical_fit
export SyntheticLogLikelihood
export L_Ctrl, L_200, L_Joint

function empirical_fit(Z::TrajectoryRandomVariable, n::Int)
    y = rand(Z, n)
    D = Distributions.fit(MvNormal, y)
end

abstract type SyntheticLogLikelihood end

struct L_NoEF{Y} <: SyntheticLogLikelihood
    data::Y
    L_NoEF(; data::Y=yobs_NoEF) where Y = new{Y}(data)
end
struct L_EF{Y} <: SyntheticLogLikelihood
    data::Y
    L_EF(; data::Y=yobs_EF) where Y = new{Y}(data)
end

struct L_Ctrl{Y} <: SyntheticLogLikelihood
    data::Y
    L_Ctrl(; data::Y) where Y = new{Y}(data)
end
struct L_200{Y} <: SyntheticLogLikelihood
    data::Y
    L_200(; data::Y) where Y = new{Y}(data)
end

struct L_Joint{Y_NoEF, Y_EF} <: SyntheticLogLikelihood
    data_NoEF::Y_NoEF
    data_EF::Y_EF
    L_Joint(; data_NoEF::Y_NoEF, data_EF::Y_EF) where Y_NoEF where Y_EF = new{Y_NoEF, Y_EF}(data_NoEF, data_EF)
end

#=
function (L::L_NoEF)(θ::ParameterVector; n=500, y=zeros(Float64, 4, n))::Float64
    Y = Y_NoEF(θ)
    try
        D = empirical_fit(y, Y)
        return sum(logpdf(D, L.data))
    catch
        return -Inf
    end
end
function (L::L_EF)(θ::ParameterVector; n=500, y=zeros(Float64, 4, n))::Float64
    Y = Y_EF(θ)
    try
        D = empirical_fit(y, Y)
        return sum(logpdf(D, L.data))
    catch
        return -Inf
    end
end
=#

function (L::L_Ctrl)(θ::ParameterVector; n=500)::Float64
    Y = Y_Ctrl(θ)
    try
        D = empirical_fit(Y,n)
        return sum(logpdf(D, L.data))
    catch
        return -Inf
    end
end
function (L::L_200)(θ::ParameterVector; n=500)::Float64
    Y = Y_200(θ)
    try
        D = empirical_fit(Y,n)
        return sum(logpdf(D, L.data))
    catch
        return -Inf
    end
end

function (L::L_Joint)(θ::ParameterVector; n=500)::Float64
    NoEF = L_Ctrl(data=L.data_NoEF)
    EF = L_200(data=L.data_EF)

    r = NoEF(θ; n=n)
    r += EF(θ; n=n)
end