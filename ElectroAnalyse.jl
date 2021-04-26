include("ElectroInference.jl")
module ElectroAnalyse

using ..ElectroInference
using Plots, StatsPlots
using LaTeXStrings
using StatsBase

export colwidth
const colwidth = 312

# Fig.1 --- See velocities
const θ_0 = Parameters([1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0], par_names)
const θ_1 = Parameters([1.0, 2.0, 0.5, 0.4, 0.4, 0.4, 0.4], par_names)
const vx = -2:.01:3
const vy = -2:0.01:2

export see_velocities
function see_velocities(; ht=1.5*colwidth, kwargs...)
    v0 = θ_0[1]
    f0 = plot(θ_0, vx, vy, title="Autonomous model", legend=:none,
        xticks=([0, v0], ["0", "v"]),
        yticks=([0, v0], ["0", "v"]),
    )

    v1 = θ_1[1]
    γ1 = θ_1[4]
    γ2 = θ_1[5]
    γ3 = θ_1[6]
    
    f1 = plot(θ_1, vx, vy, title="Electrotactic model", legend=:none,
        xticks=(v1.*[γ1-1-γ2+γ3, γ1, γ1+1+γ2+γ3], 
            [
                "γ₁v - (1+γ₂-γ₃)v",
                "γ₁v",
                "γ₁v + (1+γ₂+γ₃)v",
            ]
        ),
        yticks=([0, (1+γ2)*v1],
            [
                "0",
                "(1+γ₂)v",
            ]
        ),
        xrotation=15,
        yrotation=15,
    )
    return plot(f0, f1;
    link=:all,
    tick_direction=:out,
    layout=(2,1),
    size=(colwidth, ht),
    kwargs...)
end

export b_Ctrl, c_Ctrl
const b_Ctrl = load(:L_Ctrl, (:v, :EB, :D), fn="merged_data_post")
const c_Ctrl = load(:S_Ctrl, fn="merged_data_post")

# Fig. 2 --- posterior_NoEF

export posterior_Ctrl
function posterior_Ctrl(; ht=2*colwidth, kwargs...)
    plot(
        b_Ctrl;
        layout=(3,1),
        size = (colwidth, ht),
        seriestype=:histogram, seriescolor=1, linecolor=1,
        kwargs...
    )
end

# Fig. 2 (SI) --- posterior_NoEF_2d

export posterior_NoEF_2d
function posterior_NoEF_2d(; ht=2*colwidth, kwargs...)
    parametergrid(hcat(b_Ctrl.θ...), [1,2,3];
        size=(2*colwidth, ht),
        weights=exp.(b_Ctrl.ell),
        kwargs...
    )
end

# Fig. 3 --- compare_Ctrl

export compare_Ctrl
function compare_Ctrl(n=50; ht=1.5*colwidth, kwargs...)
    fig = plot(;
        layout=(2,1),
        legend=:none,
        ratio=:equal,
        framestyle=:origin,
        xticks=[],
        yticks=[],
        link=:all,
        size=(colwidth, ht),
        kwargs...
    )
    plot!(fig, xobs_Ctrl; subplot=1, title="Observed")

    pars = rand(b_Ctrl.θ, n)
    for t in pars
        P = P_Ctrl(t)
        plot!(fig, rand(P, save_idxs=2, saveat=5).u; subplot=2)
    end
    plot!(fig, subplot=2, title="Simulated")

    x = xlims(fig[2])[2]-20
    y = ylims(fig[2])[1]+10

    for sub in [1,2]
    plot!(fig,
        [x-200, x], [y, y];
        annotations = (x-100, y+20, Plots.text("200 μm", 7, :bottom, "helvetica")),
        color=:black,
        linewidth=4,
        subplot = sub,
        xlabel="",
        ylabel="",
        )
    end
    fig
end

# Fig.4 --- NoEF posterior distribution of simulation outputs

export predictive_Ctrl
function predictive_Ctrl(; ht=1.5*colwidth, kwargs...)
    plot(c_Ctrl;
        layout=(3,1),
        size=(colwidth, ht),
        infty=[30, 300, 1],
        seriestype=:histogram, seriescolor=2, linecolor=2,
        kwargs...,
    )
end

# Fig.2-4 --- Smush

export smush_NoEF
function smush_NoEF(n=50; kwargs...)
    l = @layout [a{0.33w} [b{0.5h}; c]]
    f2 = posterior_Ctrl(; layout=(1,3), link=:none)
    plot!(f2, subplot=1, title="(c) Polarised cell speed", ylims=:auto, xlims=:auto)
    plot!(f2, subplot=2, title="(d) Depolarisation barrier", ylims=:auto, xlims=:auto)
    plot!(f2, subplot=3, title="(e) Diffusion constant", ylims=:auto, xlims=:auto)

    f3 = compare_Ctrl(n; layout=(2,1), link=:none)
    plot!(f3, subplot=1, title="(a) Observed positions")
    plot!(f3, subplot=2, title="(b) Simulated positions")
    xlims!(f3[2], xlims(f3[1]))
    ylims!(f3[2], ylims(f3[1]))
    xlims!(f3[1], xlims(f3[2]))
    ylims!(f3[1], ylims(f3[2]))


    f4 = predictive_Ctrl(; layout=(1,3), link=:none)
    plot!(f4, subplot=1, title="(f) Time to polarise", ylims=:auto)
    plot!(f4, subplot=2, title="(g) Time to depolarise", ylims=:auto)
    plot!(f4, subplot=3, title="(h) Probability polarised", ylims=:auto, xlims=[0.8,1])

    f = plot(f3, f2, f4; size=(2*colwidth, 1*colwidth), layout=l, kwargs...)
    f
end

############# Find best model
# Fig.5 --- Model selection

export D_Joint
using Combinatorics

ps = collect(powerset([1,2,3,4]))
X_str_vec = ["$X" for X in ps]
X_str_vec[1] = "[]"
const D_Joint = Dict(
    map(X->(X, aload(:log_partition_function, :L_Joint, get_par_names(X), fn="merged_data_post")),
        ps
    )
)

J(X, μ) = D_Joint[X] - μ*(3+length(X))
function objective_function(μ; kwargs...)
    Jvec = [J(X, μ) for X in ps]
    idx = sortperm(Jvec)
    bar(
        1:16,
        Jvec[idx] .- minimum(Jvec),
        yticks=(1:16, X_str_vec[idx]),
        orientation=:horizontal,
        legend=:none,
        title=("Parametrisation cost μ = $μ"),
        ylabel="X",
        xlabel="Translated Jμ",
        kwargs...
    )
end

export see_model_selection
function see_model_selection(μ::Vararg{Any, N}; ht=N*colwidth, kwargs...) where N
    fset = objective_function.(μ)
    fig = plot(fset...;
        layout=(2,1),
        size=(colwidth,ht),
        kwargs...
    )
end
function see_model_selection(; kwargs...) 
    fig = see_model_selection(0, 2)
    plot!(fig, subplot=1, title="Model fit to data", xlabel="Translated J₀")
    plot!(fig, subplot=2, title="Fit subtract parameter cost", xlabel="Translated J₂")
    plot!(fig; kwargs...)
    fig
end

############# Use best model

export b4
const b4 = load(:L_Joint, get_par_names([4]); fn="merged_data_post")

# Fig. 6 --- EF Posteriors
export posterior_EF, posterior_compare, posterior_EF_2d
function posterior_EF(; ht=1.5*colwidth, kwargs...)
    fig = plot(
        b4,
        1:4;
        layout=(2,2),
        size = (colwidth, ht),
        kwargs...
    )
end


# Fig. 6 (SI) --- posterior_EF_2d

function posterior_EF_2d(; ht=2*colwidth, kwargs...)
    parametergrid(hcat(b4.θ...), 1:4;
        size=(2*colwidth, ht),
        kwargs...
    )
end


# Fig. 6 (SI) --- Compare posteriors
function posterior_compare(; ht=colwidth, kwargs...)
    fig = plot(
        b4,
        1:3;
        layout=(3,1),
        seriestype=:density,
        size = (colwidth, ht),
        label= "All data",
    )
    plot!(
        fig,
        b_Ctrl,
        1:3,
        seriestype=:density,
        linecolor=2,
        label= "Autonomous data",
    )
    plot!(
        fig,
        subplot=3,
        legend=true,
    )
    plot!(fig; kwargs...)
    fig
end



# Fig. 7 --- Compare EF simulation and data
export compare_Joint
function compare_Joint(B=b4, n=50; ht=1.5*colwidth, kwargs...)
    fig = plot(;
        layout=(2,2),
        legend=:none,
        ratio=:equal,
        framestyle=:origin,
        xticks=[],
        yticks=[],
        link=:all,
        size=(colwidth, ht),
        kwargs...
    )
    plot!(fig, xobs_Ctrl; subplot=1, title="Observed")
    plot!(fig, xobs_200[13:37, :] .- xobs_200[[13], :]; subplot=3, title="Observed")

    pars = rand(B.θ, n)
    for t in pars
        P = P_Ctrl(t)
        plot!(fig, rand(P, save_idxs=2, saveat=5).u; subplot=2)
    end
    plot!(fig, subplot=2, title="Simulated")

    # θbar = Parameters(mean(b4.θ), get_par_names([4]))
    # for t in Iterators.repeated(θbar, n)
    pars = rand(B.θ, n)
    for t in pars
        P = P_200(t)
        sol = rand(P, save_idxs=2, saveat=5)
        plot!(fig, sol.u[13:end] .- sol.u[13]; subplot=4)
    end
    plot!(fig, subplot=4, title="Simulated")

    x = xlims(fig[2])[2]-20
    y = ylims(fig[2])[1]+10

    for sub in 1:4
    plot!(fig,
        [x-200, x], [y, y];
        annotations = (x-100, y+20, Plots.text("200 μm", 7, :bottom, "helvetica")),
        color=:black,
        linewidth=4,
        subplot = sub,
        xlabel="",
        ylabel="",
        )
    end
    fig
end


# Fig.6-7 --- Smush

export smush_EF
function smush_EF(; kwargs...)
    
    f6 = posterior_EF(; xlims=:auto, layout=(2,2), seriestype=:histogram, seriescolor=1, linecolor=1, ylim=:auto, link=:none)
    plot!(f6, subplot=1, title="(a) Polarised cell speed")
    plot!(f6, subplot=2, title="(b) Depolarisation barrier")
    plot!(f6, subplot=3, title="(c) Diffusion constant")
    plot!(f6, subplot=4, title="(d) Polarity bias")
    
    f7 = compare_Joint(layout=(2,2), xlabel="", ylabel="", link=:none)
    plot!(f7, subplot=1, title="(e) Autonomous: observed")
    plot!(f7, subplot=2, title="(f) Autonomous: simulated")
    plot!(f7, subplot=3, title="(g) Electrotaxis: observed")
    plot!(f7, subplot=4, title="(h) Electrotaxis: simulated")

    l = @layout [a{0.5h}; b]
    f = plot(f6, f7; size=(colwidth, 1.5*colwidth), layout=l, kwargs...)

    xx = [xlims(f[j]) for j in 5:8]
    yy = [ylims(f[j]) for j in 5:8]

    _x = (minimum(L[1] for L in xx), maximum(L[2] for L in xx))
    _y = (minimum(L[1] for L in yy), maximum(L[2] for L in yy))

    for j in 5:8
        xlims!(f[j], _x)
        ylims!(f[j], _y)
    end
    f
end

# Fig. 8 --- Plot pre/post switch/stop behaviour
export θbar
const θbar = mean(b_124)

export view_step
function _pre(sol, n=5; kwargs...)
    m = pre_step_mean(sol)
    t = (pre_step_traj(sol[i]) for i in 1:n)
    fig = plot(m; c=:red, ratio=:equal, label="Mean", linewidth=3, kwargs...)
    for t_i in t
        plot!(fig, t_i; label="")
    end
    fig
end
function _post(sol, n=5; kwargs...)
    m = post_step_mean(sol)
    t = (post_step_traj(sol[i]) for i in 1:n)
    fig = plot(m; c=:red, ratio=:equal, label="Mean", linewidth=3, kwargs...)
    for t_i in t
        plot!(fig, t_i; label="")
    end
    fig
end

function view_step(θ = θbar; ht=0.75*colwidth, kwargs...)
    sol_switch = rand(P_switch(θ), 500, save_idxs=2)
    sol_stop = rand(P_stop(θ), 500, save_idxs=2)

    f1 = _pre(sol_switch, title="(a) Before switch", legend=:none)
    f2 = _post(sol_switch, title="(b) After switch", legend=:topright)
    f3 = _pre(sol_stop, title="(c) Before stop", legend=:none)
    f4 = _post(sol_stop, title="(d) After stop", legend=:none)

    fig = plot(f1,f2,f3,f4; 
        layout=(2,2),
        size=(colwidth, ht),
        xticks=[],
        yticks=[],
        xlabel="",
        ylabel="",
        framestyle=:origin,
        link=:all,
        kwargs...
    )

    x = xlims(fig[4])[2]-30
    y = ylims(fig[4])[1]+10

    for sub in [2]
        plot!(fig,
            [x-100, x], [y, y];
            annotations = (x-50, y+10, Plots.text(L"100 ~\mu m",8,:bottom)),
            label="",
            color=:black,
            linewidth=4,
            subplot = sub,
        )
    end
    fig

end

# Fig. 9 --- Polarity diagram

export coarse_polarity_diagram
function coarse_polarity_diagram(; ht=colwidth, annfontsize=10, kwargs...)
    Ω0 = Plots.Shape(Plots.partialcircle(0, 2π, 100, 0.6))
    Ωp = Plots.Shape([0,2,2], [0,2,-2])
    Ωm = Plots.Shape([0,-2,-2], [0,2,-2])
    Ωperp = Plots.Shape([0,2,-2,2,-2],[0,2,2,-2,-2])

    fig = plot(;
        legend=:none,
        ratio=:equal,
        framestyle=:box,
        lims=(-1.5,1.5),
        xticks=[],
        yticks=[],
        xlabel=L"p_x",
        ylabel=L"p_y",
        title="Coarse grained polarity",
    )
    plot!(fig, Ωp, c=1)
    plot!(fig, Ωm, c=2)
    plot!(fig, Ωperp, c=3)
    plot!(fig, Ω0, c=4)

    annotate!(fig, -0.2, 0.2, text(L"\Omega_0", annfontsize))
    annotate!(fig, 0.0, 1.0, text(L"\Omega_\perp", annfontsize))
    annotate!(fig, 0.0, -1.0, text(L"\Omega_\perp", annfontsize))
    annotate!(fig, 1.0, 0.0, text(L"\Omega_+", annfontsize))
    annotate!(fig, -1.0, 0.0, text(L"\Omega_-", annfontsize))

    scatter!(fig, [0.0], [0.0], markercolor=:black, markersize=4)
    annotate!(fig, -0.1, -0.1, text(L"0", annfontsize))
    plot!(fig, [(0,0), (0.6,0)], arrow=:closed, linecolor=:black)
    annotate!(fig, 0.3, 0.1, text(L"\bar p", annfontsize))

    plot!(fig; size=(colwidth, ht), kwargs...)
    fig
end

# Fig. 9 --- Polarity trajectories
export see_coarse_polarity
function see_coarse_polarity(θ = θbar; ht=0.75*colwidth, kwargs...)
    pbar2 = ElectroInference.pbar2(θ)
    YY = (T_depolarise(pbar2), T_pos(pbar2), T_neg(pbar2), T_perp(pbar2))

    sol_switch = rand(P_switch(θ), 500, save_idxs=1)
    sol_stop = rand(P_stop(θ), 500, save_idxs=1)
    
    fig1 = plot(0:0.1:180, YY, sol_switch, title="(a) Switching EF", labels=["Depolarised" "Positive" "Negative" "Perpendicular"], c=[4 1 2 3])
    fig2 = plot(0:0.1:180, YY, sol_stop, title="(b) Stopping EF", legend=:none, c=[4 1 2 3])

    fig = plot(fig1, fig2; layout=(1,2), size=(1.5*colwidth, ht), xlabel="Time (min)", ylabel="Proportion", kwargs...)
    fig
end

# Fig. 10 --- Posterior predictive switching stats
export predictive_step

c_Switch = load(:S_Switch)
c_Stop = load(:S_Stop)

function predictive_step(; ht=1.5*colwidth, kwargs...)
    fig = plot(;
        layout=(3,1),
        size=(colwidth, ht),
    )
    plot!(fig, c_Switch;
        infty=[180, 360, 1],
        c=1,
        label="Switch",
    )
    plot!(fig, c_Stop;
        infty=[180, 360, 1],
        c=2,
        label="Stop",
    )
    plot!(fig;
        subplot=1,
        legend=:true)
    plot!(fig; subplot=1, title="(a) Time to depolarise")
    plot!(fig; subplot=2, title="(b) Time to polarise R to L")
    plot!(fig; subplot=3, title="(c) Prob. polarised perpendicular")
    plot!(fig;
        kwargs...
    )
    fig
end


end