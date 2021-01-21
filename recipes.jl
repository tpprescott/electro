#### PARAMETER DETAILS
using Printf

const par_str = (
    L"v~\mathrm{\mu m~min}^{-1}",
    L"\Delta W_{\mathrm{on}}",
    L"\Delta W_{\mathrm{off}}",
    L"D~\mathrm{min}^{-1}",
    L"\gamma_1",
    L"\gamma_2",
    L"\gamma_3",
    L"\gamma_4",
)
const D_par_str = Dict(zip(par_names, par_str))

const par_titles = (
    "Polarised cell speed",
    "Polarisation barrier",
    "Depolarisation barrier",
    "Diffusion constant",
    "Velocity bias",
    "Speed increase",
    "Speed alignment",
    "Polarity bias",
)
const D_par_titles = Dict(zip(par_names, par_titles))

const D_par_lims = Dict(
    zip(
        par_names,
        zip(minimum.(prior_support), maximum.(prior_support)),
    )
)

# Plot a parameter set as a parameter row
@recipe function f(P::Parameters)
    P, Base.OneTo(size(P,1))
end
@recipe function f(P::Parameters, I)
    ParameterRow((P, I))
end
# Inference batch is a weighted parameter set
@recipe function f(B::InferenceBatch, I...)
    ell_max = maximum(B.ell)
    w = exp.(B.ell .- ell_max)
    weights := w
    if length(I)>1 
        seriescolor --> :Blues_6
    end
    return (B.θ, I...)
end


# Individual subplots - 1D histogram
@recipe function f(P::Parameters{Names}, I::Int) where Names
    xguide --> D_par_str[Names[I]]
    xlims --> D_par_lims[Names[I]]
    yticks --> :none
    legend := :none
    title --> D_par_titles[Names[I]]
    seriestype --> :histogram
    linecolor --> 1
    seriescolor --> 1
    normalize --> :pdf
    selectdim(P.θ, 1, I)
end

# Individual subplots - 2D scatter (as default) or histogram
@recipe function f(P::Parameters{Names}, I::Int, J::Int) where Names
    xguide --> D_par_str[Names[I]]
    yguide --> D_par_str[Names[J]]
    xlims --> D_par_lims[Names[I]]
    ylims --> D_par_lims[Names[J]]
    legend := :none
    seriestype --> :hist2d
    marker_z --> get(plotattributes, :weights, nothing)
    (selectdim(P.θ, 1, I), selectdim(P.θ, 1, J))
end

#@userplot ParameterGrid

# Define ParameterRow
@userplot ParameterRow
@recipe function f(g::ParameterRow)
    # g.args are the arguments - parameter matrix and vector of indices
    length(g.args)==2 || error("Needs two arguments: P and I")
    P, I = g.args
        
    legend := :false
    link --> :none
    layout --> (1, length(I))
    
    for (k,i) in enumerate(I)
        @series begin 
            subplot := k
            P, i
        end
    end
end
@userplot ParameterGrid
@recipe function f(g::ParameterGrid)
    # g.args are the arguments - parameter matrix and vector of indices
    length(g.args)==2 || error("Needs two arguments: P and I")
    P, I = g.args
    N = length(I)

    legend := :false
    link --> :none
    layout --> (N, N)
    
    k = 0
    for (k_i, i) in enumerate(I)
        for j in I[1:(k_i-1)]
            k += 1
            @series begin
                subplot := k
                P, j, i
            end
        end
        k += 1
        @series begin
            subplot := k
            P, i
        end
        for j in I[(k_i+1):end]
            k+=1
            @series begin
                subplot := k

                xx = quantile(selectdim(P, 1, j), [0.005, 0.5, 0.995])
                yy = quantile(selectdim(P, 1, i), [0.005, 0.5, 0.995])

                xlims := (minimum(xx), maximum(xx))
                ylims := (minimum(yy), maximum(yy))

                xticks := (xx, [@sprintf("%5.2f", xo) for xo in xx]) 
                yticks := (yy, [@sprintf("%5.2f", yo) for yo in yy])

                P, j, i
            end
        end
    end
end

#### SUMMARY DETAILS

const ss_names = (
    :T_depolarise,
    :T_polarise,
    :T_neg,
    :T_pos,
    :T_perp,
    :IsPolarised,
    :T1_lt_T2, # This should be more general but refers to a specific one I'm using in my work
)

const ss_str = (
    L"\bar T_0~\mathrm{min}",
    L"\bar T_1~\mathrm{min}",
    L"\bar T_-~\mathrm{min}",
    L"\bar T_+~\mathrm{min}",
    L"\bar T_\perp~\mathrm{min}",
    L"\bar \Pi_\infty",
    L"\bar P_{\perp \rightarrow -}",
)
const D_ss_str = Dict(zip(ss_names, ss_str))

const ss_titles = (
    "Time to depolarise",
    "Time to polarise",
    "Time to polarise R to L",
    "Time to polarise L to R",
    "Time to polarise perpendicular",
    "Proportion of cells polarised",
    "Prob. polarise perp. before R to L",
)
const D_ss_titles = Dict(zip(ss_names, ss_titles))

@recipe function f(C::ConditionalExpectation{Names}) where Names
    C, Base.OneTo(length(Names))
end
@recipe function f(C::ConditionalExpectation, I)
    ParameterRow((C, I))
end

# Individual subplots - 1D histogram
@recipe function f(C::ConditionalExpectation{Names}, I::Int) where Names
    xguide --> D_ss_str[Names[I]]
    yticks --> :none
    legend := :none
    
    title --> D_ss_titles[Names[I]]
    seriestype --> :stephist
    normalize --> :pdf

    infty_vec = get(plotattributes, :infty, fill(Inf,I))
    infty = infty_vec[I]
    idx = selectdim(C.D, 1, I).<infty

    w = exp.(C.ell .- maximum(C.ell))
    weights := w[idx]
    xlims --> (0, infty)

    @info "Posterior mass at infinity: $(sum(w[Not(idx)])/sum(w))."
    view(C.D, I, idx)
end

# Individual subplots - 2D scatter (as default) or histogram
@recipe function f(C::ConditionalExpectation{Names}, I::Int, J::Int) where Names
    xguide --> D_ss_str[Names[I]]
    yguide --> D_ss_str[Names[J]]
    legend := :none
    seriestype --> :hist2d

    infty_vec = get(plotattributes, :infty, fill(Inf, max(I,J)))
    infty_I = infty_vec[I]
    infty_J = infty_vec[J]
    idx = (selectdim(C.D, 1, I).<infty_I) .* (selectdim(C.D, 1, J).<infty_J)

    w = exp.(C.ell .- maximum(C.ell))
    weights := w[idx]
    xlims --> (0, infty_I)
    ylims --> (0, infty_J)

    @info "Posterior mass at infinity: $(sum(w[Not(idx)])/sum(w))."
    marker_z --> get(plotattributes, :weights, nothing)
    (selectdim(C.D, 1, I), selectdim(C.D, 1, J))
end


##### SEE VELOCITY
@recipe function f(θ::ParameterVector{Names}, vx, vy) where Names
    θ_nt = NamedTuple{Names}(θ.θ)
    VelocityDistribution((θ_nt, vx, vy))
end
@recipe function f(θ::NamedTuple, vx, vy)
    β, pbar2 = _map_barriers_to_coefficients(θ.EB_on, θ.EB_off)
    W(p) = W_poly(β, pbar2)(abs2(p))
    
    polarity_density(p) = exp(θ.γ4*real(p) - W(p))
    function get_pol(v)
        v_EM = θ.γ1*θ.v*complex(1)
        v_cell = v - v_EM
        if iszero(v_cell)
            return v_cell
        else
            p_hat = v_cell/abs(v_cell)
            p = v_cell / (θ.v * (1 + θ.γ2 + θ.γ3*real(p_hat)))
            return p
        end
    end

    velocity_density(x, y) = polarity_density(get_pol(complex(x, y)))
    (vx, vy, velocity_density)
end

@userplot VelocityDistribution
@recipe function f(g::VelocityDistribution)
    # g.args are the arguments - parameter matrix and vector of indices
    length(g.args)==3 || error("Needs three arguments: θ (named tuple) and vx and vy")
    θ, vx, vy = g.args
        
    colorbar --> :none
    xlims --> (minimum(vx), maximum(vx))
    ylims --> (minimum(vy), maximum(vy))
    aspect_ratio --> :equal
    legend := :none

    @series begin 
        seriescolor --> :Blues
        seriestype --> :heatmap
        θ, vx, vy
    end

    @series begin
        seriestype := :hline
        seriescolor := :black
        [0]
    end

    @series begin
        seriestype := :vline
        seriescolor := :black
        [0]
    end

end

#### View trajectories against data

@recipe function f(θ::ParameterVector, ::Type{NoEF})
    P = P_NoEF(θ)
    sol = rand(P, 50)
    xsim = observation_filter(sol)
    CompareTrajectories((xobs_NoEF, xsim))
end

@recipe function f(θ::ParameterVector, ::Type{ConstantEF})
    P = P_EF(θ)
    sol = rand(P, 50)
    xsim = observation_filter(sol)
    CompareTrajectories((xobs_EF, xsim))
end

@userplot CompareTrajectories
@recipe function f(g::CompareTrajectories)
    xobs, xsim = g.args
    
    layout --> (2,1)
    legend := :none
    framestyle --> :origin
    aspect_ratio := :equal
    link := :all
    titlefontsize --> 11
    tickfontsize --> 8
    xguide --> ""
    yguide --> ""
    xticks --> []
    yticks --> []

    @series begin
        subplot := 1
        title := "Observed positions"
        xobs
    end

    @series begin
        subplot := 2
        title := "Simulated positions"
        xsim
    end

end

@recipe function f(T, YY::NTuple{N, HittingTime}, sol::EnsembleSolution) where N
    D = zeros(length(T), N)
    for (i,t) in enumerate(T)
        D[i,:] .= _proportion(t, YY, sol)
    end
    xticks := 0:90:maximum(T)
    (T, D)
end
