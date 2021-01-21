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

function Fig1(fn="Compare_Velocity.pdf", pth=pth)
    fig = see_velocities(; ht=1.5*colwidth, sizes..., tickfontsize=7, yrotation=0, framestyle=:box, ytickhfontalign=:right, tickfontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end

function Fig2(fn="NoEF_Posterior.pdf", pth=pth)
    fig = posterior_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigA(fn="NoEF_Posterior_Grid.pdf", pth=pth)
    fig = posterior_NoEF_2d(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig3(fn="NoEF_Compare.pdf", pth=pth)
    fig = compare_NoEF(; sizes..., ht=1.5*colwidth)
    savefig(fig, fn, pth)
    return fig
end

function Fig4(fn="NoEF_Predictive.pdf", pth=pth)
    fig = predictive_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig24(fn="NoEF_Smush.pdf", pth=pth)
    fig = smush_NoEF(; sizes..., guidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end

function Fig5(fn="Model_Selection.pdf", pth=pth)
    fig = see_model_selection(; sizes..., xguidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end

function Fig6(fn="EF_Posterior.pdf", pth=pth)
    fig = posterior_EF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigB(fn="EF_Posterior_Grid.pdf", pth=pth)
    fig = posterior_EF_2d(; sizes..., guidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end

function SIFigC(fn="Posterior_Compare.pdf", pth=pth)
    fig = posterior_compare(; sizes..., seriestype=:stephist, size=(1.5*colwidth, colwidth), legendfontsize=8, guidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end

function Fig7(fn="EF_Compare.pdf", pth=pth)
    fig = compare_EF(; ht=1.5*colwidth, sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig67(fn="EF_Smush.pdf", pth=pth)
    fig = smush_EF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig8(fn="Predict_Displacement.pdf", pth=pth)
    fig = view_step(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig9a(fn="Polarity_Diagram.pdf", pth=pth)
    fig = coarse_polarity_diagram(; sizes..., labelfontsize=10, titlefontsize=10)
    savefig(fig, fn, pth)
    return fig
end

function Fig9(fn="Predict_Polarity.pdf", pth=pth)
    fig = see_coarse_polarity(; sizes..., legendfontsize=6)
    savefig(fig, fn, pth)
    return fig
end

function Fig10(fn="EF_Predictive.pdf", pth=pth)
    fig = predictive_step(; sizes..., xguidefontfamily=:match)
    savefig(fig, fn, pth)
    return fig
end