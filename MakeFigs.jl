include("ElectroAnalyse.jl")
import Plots.savefig
using .ElectroAnalyse

sizes = (titlefontsize=11, labelfontsize=8)

savefig(fig, fn, pth) = savefig(fig, pth*fn)
pth = "figs/"

function Fig1(fn="Compare_Velocity.pdf", pth=pth)
    fig = see_velocities(; ht=1.5*colwidth, sizes..., tickfontsize=7, yrotation=0, framestyle=:box, ytickhfontalign=:right)
    savefig(fig, fn, pth)
    return fig
end

function Fig2(fn="NoEF_Posterior.pdf", pth=pth)
    fig = posterior_NoEF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigA(fn="NoEF_Posterior_Grid.pdf", pth=pth)
    fig = posterior_NoEF_2d(; titlefontsize=8, labelfontsize=8, tickfontsize=6)
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

function Fig5(fn="Model_Selection.pdf", pth=pth)
    fig = see_model_selection(; sizes..., tickfontsize=6)
    savefig(fig, fn, pth)
    return fig
end

function Fig6(fn="EF_Posterior.pdf", pth=pth)
    fig = posterior_EF(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function SIFigB(fn="EF_Posterior_Grid.pdf", pth=pth)
    fig = posterior_EF_2d(; titlefontsize=8, labelfontsize=8, tickfontsize=6)
    savefig(fig, fn, pth)
    return fig
end

function SIFigC(fn="Posterior_Compare.pdf", pth=pth)
    fig = posterior_compare(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig7(fn="EF_Compare.pdf", pth=pth)
    fig = compare_EF(; ht=1.5*colwidth, sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig8(fn="Predict_Displacement.pdf", pth=pth)
    fig = view_step(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig9a(fn="Polarity_Diagram.pdf", pth=pth)
    fig = coarse_polarity_diagram(; sizes...)
    savefig(fig, fn, pth)
    return fig
end

function Fig9(fn="Predict_Polarity.pdf", pth=pth)
    fig = see_coarse_polarity(; sizes..., legendfontsize=6, ylabel="Proportion")
    savefig(fig, fn, pth)
    return fig
end

function Fig10(fn="EF_Predictive.pdf", pth=pth)
    fig = predictive_step(; sizes...)
    savefig(fig, fn, pth)
    return fig
end