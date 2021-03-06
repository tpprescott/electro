include("ElectroAnalyse.jl")
import Plots.savefig
using .ElectroAnalyse, LaTeXStrings

sizes = (
    titlefontfamily="helvetica",
    tickfontfamily="helvetica",
    guidefontfamily="helvetica",
    legendfontfamily="helvetica",
    titlefontsize=8,
    tickfontsize=6,
    guidefontsize=8,
    legendfontsize=7,
)

savefig(fig, fn, pth) = savefig(fig, pth*fn)
pth = "figs/"

function Fig1(fn="Compare_Velocity.svg", pth=pth)
    fig = see_velocities(; ht=1.5*colwidth, sizes..., tickfontsize=7, yrotation=0, framestyle=:box, ytickhfontalign=:right)
    savefig(fig, fn, pth)
    return fig
end

function Fig2(fn="NoEF_Posterior.svg", pth=pth)
    fig = posterior_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigA(fn="NoEF_Posterior_Grid.svg", pth=pth)
    fig = posterior_NoEF_2d(; sizes...)
    for idx in 1:4:9
        plot!(fig, subplot=idx, xlims=:auto)
    end
    savefig(fig, fn, pth)
    return fig
end

function Fig3(fn="NoEF_Compare.svg", pth=pth)
    fig = compare_NoEF(; sizes..., ht=1.5*colwidth)
    savefig(fig, fn, pth)
    return fig
end

function Fig4(fn="NoEF_Predictive.svg", pth=pth)
    fig = predictive_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig24(fn="NoEF_Smush.svg", pth=pth)
    fig = smush_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig5(fn="Model_Selection.svg", pth=pth)
    fig = see_model_selection(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig6(fn="EF_Posterior.svg", pth=pth)
    fig = posterior_EF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigB(fn="EF_Posterior_Grid.svg", pth=pth)
    fig = posterior_EF_2d(; sizes...)
    for (idx, xticks) in zip(1:5:16, [.75:.25:1.5, 0:0.02:0.06, 0:0.03:0.18, 0:0.3:1.2])
        plot!(fig, subplot=idx, xlims=:auto)
    end
    savefig(fig, fn, pth)
    return fig
end

function SIFigC(fn="Posterior_Compare.svg", pth=pth)
    fig = posterior_compare(; sizes..., size=(colwidth, 1.5*colwidth), xlims=:auto)
    savefig(fig, fn, pth)
    return fig
end

function Fig7(fn="EF_Compare.svg", pth=pth)
    fig = compare_Joint(; ht=1.5*colwidth, sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig67(fn="EF_Smush.svg", pth=pth)
    fig = smush_EF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig89(fn="Switch_Smush.svg", pth=pth)
    fig, x = smush_Switch(; sizes...)
    savefig(fig, fn, pth)
    return fig, x
end

#=
function Fig8(fn="Predict_Displacement.svg", pth=pth)
    fig = view_step(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig9a(fn="Polarity_Diagram.svg", pth=pth)
    fig = coarse_polarity_diagram(; sizes..., labelfontsize=10, titlefontsize=10)
    savefig(fig, fn, pth)
    return fig
end

function Fig9(fn="Predict_Polarity.svg", pth=pth)
    fig = see_coarse_polarity(; sizes..., legendfontsize=6)
    savefig(fig, fn, pth)
    return fig
end

function Fig10(fn="EF_Predictive.svg", pth=pth)
    fig = predictive_step(; sizes..., xguidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end
=#