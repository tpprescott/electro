include("ElectroInference.jl")
module ElectroAnalyse

using ..ElectroInference
using Plots
using LaTeXStrings

const colwidth = 312

# Fig.1 --- See velocities
const θ_0 = Parameters([1.0, 2.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0], par_names)
const θ_1 = Parameters([1.0, 2.0, 2.0, 0.5, 0.4, 0.4, 0.4, 0.4], par_names)
const vx = -2:.01:3
const vy = -2:0.01:2

export see_velocities
function see_velocities(; ht=600, kwargs...)
    v0 = θ_0[1]
    f0 = plot(θ_0, vx, vy, title="Autonomous model", legend=:none,
        xticks=([0, v0], [L"0", L"v"]),
        yticks=([0, v0], [L"0", L"v"]),
    )

    v1 = θ_1[1]
    γ1 = θ_1[5]
    γ2 = θ_1[6]
    γ3 = θ_1[7]
    
    f1 = plot(θ_1, vx, vy, title="Electrotactic model", legend=:none,
        xticks=(v1.*[γ1-1-γ2+γ3, γ1, γ1+1+γ2+γ3], 
            [
                L"\gamma_1 v - (1+\gamma_2 - \gamma_3)v",
                L"\gamma_1 v",
                L"\gamma_1 v + (1 + \gamma_2 + \gamma_3)v",
            ]
        ),
        yticks=([0, (1+γ2)*v1],
            [
                L"0",
                L"1 + \gamma_2 v",
            ]
        ),
        xrotation=45,
    )
    return plot(f0, f1;
    link=:all,
    tick_direction=:out,
    layout=(2,1),
    titlefontsize=11,
    tickfontsize=8,
    size=(colwidth, ht),
    kwargs...)
end

export b_NoEF, c_NoEF
const b_NoEF = load(:L_NoEF, (:v, :EB_on, :EB_off, :D), fn="electro_data")
const c_NoEF = load(:S_NoEF, fn="electro_data")

# Fig. 2 --- posterior_NoEF

export posterior_NoEF
function posterior_NoEF(; ht=2*colwidth, kwargs...)
    plot(
        b_NoEF;
        layout=(4,1),
        size = (colwidth, ht),
        titlefontsize=11,
        tickfontsize=8,
        kwargs...
    )
end

# Fig. 2 (SI) --- posterior_NoEF_2d

export posterior_NoEF_2d
function posterior_NoEF_2d(; ht=2*colwidth, kwargs...)
    parametergrid(b_NoEF.θ, [1,2,3,4];
        size=(2*colwidth, ht),
        titlefontsize=8,
        labelfontsize=8,
        tickfontsize=6,
        kwargs...
    )
end

# Fig. 3 --- compare_NoEF

export compare_NoEF
function compare_NoEF(θ = mean(b_NoEF); ht=2*colwidth, kwargs...)
    plot(θ, NoEF; 
        size = (colwidth, ht),
        kwargs...,
    )
end

# Fig.4 --- NoEF posterior distribution of simulation outputs

function predictive_NoEF(; ht=2*colwidth, kwargs...)
    plot(c_NoEF;
        layout=(3,1),
        size=(colwidth, ht),
        kwargs...,
    )
end

# Fig. 9 --- Polarities

function see_coarse_polarity(; kwargs...)
    Ω0 = Plots.Shape(Plots.partialcircle(0, 2π, 100, 0.6))
    Ωp = Plots.Shape([0,2,2], [0,2,-2])
    Ωm = Plots.Shape([0,-2,-2], [0,2,-2])
    Ωperp = Plots.Shape([0,2,-2,2,-2],[0,2,2,-2,-2])

    fig = plot(;
        legend=:none,
        ratio=:equal,
        lims=(-1.5,1.5),
        title="Coarse-grained polarity",
        xlabel=L"p_x",
        ylabel=L"p_y",
        titlefontsize=11pt,
    )
    plot!(fig, Ωp, c=1)
    plot!(fig, Ωm, c=2)
    plot!(fig, Ωperp, c=3)
    plot!(fig, Ω0, c=4)

    annotate!(fig, 0.0, 0.0, L"\Omega_0")
    annotate!(fig, 0.0, 1.0, L"\Omega_\perp")
    annotate!(fig, 0.0, -1.0, L"\Omega_\perp")
    annotate!(fig, 1.0, 0.0, L"\Omega_+")
    annotate!(fig, -1.0, 0.0, L"\Omega_-")

    plot!(fig; kwargs...)
    fig
end

end