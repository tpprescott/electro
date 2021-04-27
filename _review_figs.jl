include("ElectroInference.jl")
using .ElectroInference
using Plots, StatsPlots, CSV
using StatsBase
using LaTeXStrings
using Distributions

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

B1 = load(:L_Ctrl, par_names[1:3], fn="replicate_1_post")
B2 = load(:L_Ctrl, par_names[1:3], fn="replicate_2_post")
B = load(:L_Ctrl, par_names[1:3], fn="merged_data_post")

###########
# Displacements
disp_ctrl_1 = diff(xobs_Ctrl_1, dims=1)
disp_ctrl_2 = diff(xobs_Ctrl_2, dims=1)

function preliminary_analysis(disp_ctrl_1=disp_ctrl_1, disp_ctrl_2=disp_ctrl_2; kwargs...)
  # Polar coordinates
  mod_ctrl_1 = abs.(disp_ctrl_1)
  mod_ctrl_2 = abs.(disp_ctrl_2)
  arg_ctrl_1 = angle.(disp_ctrl_1)
  arg_ctrl_2 = angle.(disp_ctrl_2)

  # Plot some displacement lengths distributions for all cells
  fig1A = density(mod_ctrl_1; legend=:none, xlims=[0, Inf], xlabel="Displacement (μm)", title="(a) Control 1: displacement distances (5min)", yticks=[], size=(360,240), sizes...)
  fig1B = density(mod_ctrl_2; legend=:none, xlims=[0, Inf], xlabel="Displacement (μm)", title="(b) Control 2: displacement distances (5min)", yticks=[], size=(360,240), sizes...)

  # Plot some displacement angles over time (see persistence and drift?)
  #fig1C = plot((1:60).*5, arg_ctrl_1[:,3:10:end]; legend=:none, ylim=[-pi,pi], yticks=([-pi, -pi/2, 0, pi/2, pi], ["-π", "-π/2", "0", "π/2", "π"]), xlabel="Interval End (min)", ylabel="Displacement angle", title="(c) Control 1: displacement angles", size=(360,240), sizes..., yguidefontfamily=:auto)
  #fig1D = plot((1:60).*5, arg_ctrl_2[:,3:10:end]; legend=:none, ylim=[-pi,pi], yticks=([-pi, -pi/2, 0, pi/2, pi], ["-π", "-π/2", "0", "π/2", "π"]), xlabel="Interval End (min)", ylabel="Displacement angle", title="(d) Control 2: displacement angles", size=(360,240), sizes...)

  # Angle autocorrelation? Make more continuous over the jump, use sin and cos
  fig1E = plot(0:12, autocor(sin.(arg_ctrl_1), 0:12); xticks=(0:12, (0:12).*5), ylims=[-1,1], legend=:none, xlabel="Lag (min)", ylabel="Autocorrelation in sin(angle)", title="(c) Control 1: Direction autocorrelation", size=(360,240), sizes...)
  fig1F = plot(0:12, autocor(sin.(arg_ctrl_2), 0:12); xticks=(0:12, (0:12).*5), ylims=[-1,1], legend=:none, xlabel="Lag (min)", ylabel="Autocorrelation in sin(angle)", title="(d) Control 2: Direction autocorrelation", size=(360,240), sizes...)

  #fig1 = plot(fig1A, fig1B, fig1C, fig1D, fig1E, fig1F, layout=(3,2), size=(720, 720))
  fig1 = plot(fig1A, fig1B, fig1E, fig1F, layout=(2,2), size=(720, 480))
  plot!(fig1; kwargs...)
end
fig1 = preliminary_analysis()

function simulate_preliminary_analysis(disp_ctrl_1=disp_ctrl_1, disp_ctrl_2=disp_ctrl_2, B1=B1, B2=B2; kwargs...)
  tset_1 = rand(B1.θ, 27)
  tset_2 = rand(B2.θ, 26)

  disp_sim_1 = similar(disp_ctrl_1)
  disp_sim_2 = similar(disp_ctrl_2)

  for (i,t) in enumerate(tset_1)
    sol = rand(P_Ctrl(t), saveat=5, save_idxs=2).u
    selectdim(disp_sim_1, 2, i) .= diff(sol)
  end
  for (i,t) in enumerate(tset_2)
    sol = rand(P_Ctrl(t), saveat=5, save_idxs=2).u
    selectdim(disp_sim_2, 2, i) .= diff(sol)
  end

  fig2 = preliminary_analysis(disp_sim_1, disp_sim_2)
  plot!(fig2; kwargs...)
end
fig2 = simulate_preliminary_analysis()


###########
# See data summary statistics and simulate posterior predictive distributions of summary statistics
function see_summaries(B=B, B1=B1, B2=B2; kwargs...)

  g(p) = rand(Y_Ctrl(p.θ), 10)
  Y1 = map(g, B1);
  Y2 = map(g, B2);
  Y = map(g, B);

  # Y0dims = [vcat((y[i,:] for y in Y0)...) for i in 1:4];
  Y1dims = [vcat((y[i,:] for y in Y1)...) for i in 1:4];
  Y2dims = [vcat((y[i,:] for y in Y2)...) for i in 1:4];
  Ydims = [vcat((y[i,:] for y in Y)...) for i in 1:4];

  # f0 = plot(; layout=(2,2), legend=:none)
  fig3 = plot(; layout=(2,2), legend=:none)
  titles = ["Horizontal displacement", "Overall displacement", "Path length", "Interval variability"]
  labels = ["Displacement (μm)", "Displacement (μm)", "Path length (μm)", "Standard deviation (μm)"]

  for i in 1:4
  #  density!(f0, Y0dims[i], subplot=i, weights=W0, label="Post. pred.", seriescolor=1)
    density!(fig3, Y1dims[i], subplot=i, label="Post. pred. 1", seriescolor=1)
    density!(fig3, Y2dims[i], subplot=i, label="Post. pred. 2", seriescolor=2)
    density!(fig3, Ydims[i], subplot=i, label="Post. pred.", seriescolor=3)
  #  density!(f0, yobs_NoEF[i,:], subplot=i, label="Data", seriescolor=3, linestyle=:dash, title=titles[i])
    
    yy0, yy1 = ylims(fig3[i])
    scatter!(fig3, yobs_Ctrl_1[i,:], fill(yy1/3, size(yobs_Ctrl_1, 2)), subplot=i, label="Data 1", seriescolor=4, markershape=:x)
    scatter!(fig3, yobs_Ctrl_2[i,:], fill(2*yy1/3, size(yobs_Ctrl_2, 2)), subplot=i, label="Data 2", seriescolor=5, markershape=:x, title=titles[i], yticks=[], xlabel=labels[i])  
  end

  F1 = fit(MvNormal, hcat(Y1dims...)')
  F2 = fit(MvNormal, hcat(Y2dims...)')
  F = fit(MvNormal, hcat(Ydims...)')

  xvalidate = [sum(logpdf(f, y)) for (f,y) in Iterators.product([F1,F2,F],[yobs_Ctrl_1, yobs_Ctrl_2])]
  return fig3, xvalidate
end
fig3, xvalidate = see_summaries()

###########
# Compare posteriors

fig4 = plot(B1; xlims=:auto, size=(2*colwidth,0.5*colwidth), seriescolor=1, seriestype=:density, label="Replicate 1", sizes...)
plot!(fig4, B2; xlims=:auto, seriestype=:density, linecolor=2, label="Replicate 2")
plot!(fig4, B; xlims=:auto, seriestype=:density, linecolor=3, label="Combined")
plot!(fig4, subplot=3, legend=:topright)