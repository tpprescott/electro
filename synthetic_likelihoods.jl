export empirical_fit
export SyntheticLogLikelihood
export L_NoEF, L_EF, L_Joint

function empirical_fit(Z::TrajectoryRandomVariable, n::Int)
    y = rand(Z, n)
    D = Distributions.fit(MvNormal, y)
end
function empirical_fit(y, Z::TrajectoryRandomVariable)
    Distributions.rand!(Z, y)
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

struct L_Joint{Y_NoEF, Y_EF} <: SyntheticLogLikelihood
    data_NoEF::Y_NoEF
    data_EF::Y_EF
    L_Joint(; data_NoEF::Y_NoEF=yobs_NoEF, data_EF::Y_EF=yobs_EF) where Y_NoEF where Y_EF = new{Y_NoEF, Y_EF}(data_NoEF, data_EF)
end

function (L::L_NoEF)(θ::ParameterVector; n=500, y=zeros(Float64, 4, n))::Float64
    Y = Y_NoEF(θ)
    D = empirical_fit(y, Y)
    return sum(logpdf(D, L.data))
end
function (L::L_EF)(θ::ParameterVector; n=500, y=zeros(Float64, 4, n))::Float64
    Y = Y_EF(θ)
    D = empirical_fit(y, Y)
    return sum(logpdf(D, L.data))
end
function (L::L_Joint)(θ::ParameterVector; n=500, y=zeros(Float64, 4, n))::Float64
    NoEF = L_NoEF(data=L.data_NoEF)
    EF = L_EF(data=L.data_EF)

    r = NoEF(θ, y=y)
    r += EF(θ, y=y)
end