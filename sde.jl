NOISEFORM = [complex(1.0), complex(0.0)]

function _map_barriers_to_coefficients(EB)
    β = 4*EB
    return β
end

W_poly(β) = (β/4)*Polynomial([0, -2, 1])
∇W_poly(β) = β*Polynomial([-1, 1])

# Easy case - for zero input
function SDEdrift(emf::NoEF;
    v, EB, D,
    kwargs...)

    β = _map_barriers_to_coefficients(EB)
    ∇W = ∇W_poly(β)
    
    dpol(pol) = -D * (∇W(abs2(pol))*pol)
    vel(pol) = v*pol

    function f!(du, u, p, t)
        du[1] = dpol(u[1])
        du[2] = vel(u[1])
    end
    return f!
end

# Harder case - for non-zero input
dotprod(z1,z2) = real(z1)*real(z2) + imag(z1)*imag(z2)
function SDEdrift(emf::AbstractEMField;
    v, EB, D,
    γ1=0.0, γ2=0.0, γ3=0.0, γ4=0.0,
    kwargs...)

    β = _map_barriers_to_coefficients(EB)
    ∇W = ∇W_poly(β)

    dpol(pol, inp) = -D*(∇W(abs2(pol))*pol - γ4*inp)
    function vel(pol,inp) 
        x = γ1*v*inp
        pol==0 && (return x)
        x += v*pol*(1 + γ2*abs(inp) + γ3*dotprod(inp,pol)/abs(pol))
        return x
    end

    function f!(du, u, p, t)
        du[1] = dpol(u[1], emf(t))
        du[2] = vel(u[1], emf(t))
    end
    return f!
end

function SDEnoise(
    ;
    D,
    kwargs...)

    σ = sqrt(2*D)

    function g!(du, u, p, t)
        du[1] = σ
        du[2] = 0
    end

    return g!
end