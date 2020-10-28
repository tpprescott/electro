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


@recipe function f(P::Parameters)
    P, Base.OneTo(size(P,1))
end

@recipe function f(P::Parameters, I)
    ParameterRow((P, I))
end

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

@recipe function f(B::InferenceBatch, I...; weighted=false, temperature=1.0, accept_fun = B->isaccepted(B, temp=temperature) )
    if weighted
        weights = B.log_π - B.log_q + temperature*B.log_sl
        weights .-= maximum(weights)
        broadcast!(exp, weights, weights)
        weights := weights
        if length(I)>1 
            c --> :Blues_6
        end
        return (B.θ, I...)
    else
        I = accept_fun(B)
        return (B[I], I...)
    end
end



