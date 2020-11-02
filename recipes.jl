const par_str = (
    L"v~\mathrm{\mu m~min}^{-1}",
    L"\Delta W_{on}",
    L"\Delta W_{off}",
    L"D~\mathrm{min}^{-1}",
    L"\gamma_1",
    L"\gamma_2",
    L"\gamma_3",
    L"\gamma_4",
)

const par_titles = (
    "Polarised cell speed",
    "Energy barrier to polarisation",
    "Energy barrier to depolarisation",
    "Diffusion constant",
    "Velocity bias",
    "Speed increase",
    "Speed alignment",
    "Polarity bias",
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


# Individual subplots - 1D historgram
@recipe function f(P::Parameters, I::Int)
    xguide --> par_str[I]
    xlims --> (minimum(prior_support[I]), maximum(prior_support[I]))
    yticks --> :none
    legend := :none
    title --> par_titles[I]
    seriestype --> :histogram
    normalize --> :pdf
    selectdim(P.θ, 1, I)
end

# Individual subplots - 2D scatter (as default) or histogram
@recipe function f(P::Parameters, I::Int, J::Int)
    xguide --> par_str[I]
    yguide --> par_str[J]
    xlims --> (minimum(prior_support[I]), maximum(prior_support[I]))
    ylims --> (minimum(prior_support[J]), maximum(prior_support[J]))
    legend := :none
    seriestype --> :scatter
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
#            seriestype --> :histogram
            P, i
        end
    end
end

