include("ElectroInference.jl")
module ElectroAnalyse

using ..ElectroInference
using Plots
using LaTeXStrings

export colwidth
const colwidth = 312

# Fig.1 --- See velocities
const θ_0 = Parameters([1.0, 2.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0], par_names)
const θ_1 = Parameters([1.0, 2.0, 2.0, 0.5, 0.4, 0.4, 0.4, 0.4], par_names)
const vx = -2:.01:3
const vy = -2:0.01:2

export see_velocities
function see_velocities(; ht=1.5*colwidth, kwargs...)
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
                L"(1 + \gamma_2) v",
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
        kwargs...
    )
end

# Fig. 2 (SI) --- posterior_NoEF_2d

export posterior_NoEF_2d
function posterior_NoEF_2d(; ht=2*colwidth, kwargs...)
    parametergrid(b_NoEF.θ, [1,2,3,4];
        size=(2*colwidth, ht),
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

export predictive_NoEF
function predictive_NoEF(; ht=2*colwidth, kwargs...)
    plot(c_NoEF;
        layout=(3,1),
        size=(colwidth, ht),
        infty=[180, 90, 1],
        kwargs...,
    )
end

############# Find best model
# Fig.5 --- Model selection

export D_EF
X_str_vec = ["$X" for X in combination_powerset]
X_str_vec[1] = "[]"
const D_EF = Dict(
    map(X->(X, aload(:log_partition_function, :L_EF, get_par_names(X))),
        combination_powerset
    )
)

J(X, μ) = D_EF[X] - μ*(4+length(X))
function objective_function(μ; kwargs...)
    Jvec = [J(X, μ) for X in combination_powerset]
    idx = sortperm(Jvec)
    bar(
        1:16,
        Jvec[idx] .- minimum(Jvec),
        yticks=(1:16, X_str_vec[idx]),
        orientation=:horizontal,
        legend=:none,
        title=latexstring("\\mathrm{Parametrisation~cost}~\\mu = $μ"),
        ylabel=L"X",
        xlabel=latexstring("\\mathrm{Translated}~J_$μ"),
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
see_model_selection(; kwargs...) = see_model_selection(0, 2; kwargs...)

############# Use best model

export b_124
const b_124 = load(:L_EF, get_par_names([1,2,4]); fn="electro_data")

# Fig. 6 --- EF Posteriors
export posterior_EF, posterior_compare, posterior_EF_2d
function posterior_EF(; ht=1.5*colwidth, kwargs...)
    fig = plot(
        b_124,
        5:7;
        layout=(3,1),
        size = (colwidth, ht),
        kwargs...
    )
end


# Fig. 6 (SI) --- posterior_EF_2d

function posterior_EF_2d(; ht=3*colwidth, kwargs...)
    parametergrid(b_124.θ, 1:7;
        size=(3*colwidth, ht),
        kwargs...
    )
end


# Fig. 6 (SI) --- Compare posteriors
function posterior_compare(; ht=2*colwidth, kwargs...)
    fig = plot(
        b_124,
        1:4;
        layout=(4,1),
        size = (colwidth, ht),
        label= L"\pi(\theta | \mathbf{x}_\mathrm{NoEM}, \mathbf{x}_\mathrm{EM})",
    )
    plot!(
        fig,
        b_NoEF,
        1:4,
        linestyle=:dot,
        label= L"\pi(\theta | \mathbf{x}_\mathrm{NoEM})",
    )
    plot!(
        fig,
        subplot=1,
        legend=true,
    )
    plot!(fig; kwargs...)
    fig
end



# Fig. 7 --- Compare EF simulation and data
export compare_EF
function compare_EF(θ = mean(b_124); ht=2*colwidth, kwargs...)
    plot(θ, ConstantEF; 
        size = (colwidth, ht),
        kwargs...,
    )
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

function view_step(θ = θbar; ht=2.5*colwidth, kwargs...)
    sol_switch = rand(P_switch(θ), 500, save_idxs=2)
    sol_stop = rand(P_stop(θ), 500, save_idxs=2)

    f1 = _pre(sol_switch, title=L"\mathbf{u}_\mathrm{switch}~:~0<t<90~\mathrm{min}")
    f2 = _post(sol_switch, title=L"\mathbf{u}_\mathrm{switch}~:~90<t<180~\mathrm{min}", legend=:none)
    f3 = _pre(sol_stop, title=L"\mathbf{u}_\mathrm{stop}~:~0<t<90~\mathrm{min}", legend=:none)
    f4 = _post(sol_stop, title=L"\mathbf{u}_\mathrm{stop}~:~90<t<180~\mathrm{min}", legend=:none)

    plot(f1,f2,f3,f4; 
        layout=(4,1),
        size=(colwidth, ht),
        xlabel=L"x",
        ylabel=L"y",
        framestyle=:box,
        link=:all,
        kwargs...
    )
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
        lims=(-1.5,1.5),
        title="Coarse-grained polarity",
        ticks=([0.0, 0.6], [L"0", L"\bar p"]),
        tick_direction = :out,
    )
    plot!(fig, Ωp, c=1)
    plot!(fig, Ωm, c=2)
    plot!(fig, Ωperp, c=3)
    plot!(fig, Ω0, c=4)

    annotate!(fig, 0.0, 0.0, text(L"\Omega_0", annfontsize))
    annotate!(fig, 0.0, 1.0, text(L"\Omega_\perp", annfontsize))
    annotate!(fig, 0.0, -1.0, text(L"\Omega_\perp", annfontsize))
    annotate!(fig, 1.0, 0.0, text(L"\Omega_+", annfontsize))
    annotate!(fig, -1.0, 0.0, text(L"\Omega_-", annfontsize))

    plot!(fig; size=(colwidth, ht), kwargs...)
    fig
end

# Fig. 9 --- Polarity trajectories
export see_coarse_polarity
function see_coarse_polarity(θ = θbar; ht=1.5*colwidth, kwargs...)
    pbar2 = ElectroInference.pbar2(θ)
    YY = (T_depolarise(pbar2), T_pos(pbar2), T_neg(pbar2), T_perp(pbar2))

    sol_switch = rand(P_switch(θ), 500, save_idxs=1)
    sol_stop = rand(P_stop(θ), 500, save_idxs=1)
    
    fig1 = plot(0:0.1:180, YY, sol_switch, title=L"\mathbf{u}_\mathrm{switch}", labels=["Depolarised" "Positive" "Negative" "Perpendicular"], c=[4 1 2 3])
    fig2 = plot(0:0.1:180, YY, sol_stop, title=L"\mathbf{u}_\mathrm{stop}", legend=:none, c=[4 1 2 3])

    fig = plot(fig1, fig2; layout=(2,1), size=(colwidth, ht), kwargs...)
    fig
end

# Fig. 10 --- Posterior predictive switching stats
export predictive_step

c_Switch = load(:S_Switch)
c_Stop = load(:S_Stop)

function predictive_step(; ht=2*colwidth, kwargs...)
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
    plot!(fig;
        kwargs...
    )
    fig
end


end