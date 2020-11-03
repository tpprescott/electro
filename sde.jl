NOISEFORM = [complex(1.0), complex(0.0)]

function _map_barriers_to_coefficients(EB_on, EB_off)
    pbar2 = get_pbar2(EB_on, EB_off)
    β = get_β(EB_on, pbar2)
    return β, pbar2
end

using Polynomials: Polynomial, roots, fromroots
F(R) = Polynomial([-R, 3*R, -3*(R-1), R-1])
validp2(p2) = isreal(p2) && (0<real(p2)<1)
function get_pbar2(EB_on, EB_off)
    R = EB_on/EB_off
    p2vec = filter(validp2, roots(F(R)))
    length(p2vec)==1 || error("Too many (or not enough) valid roots found!")
    return real(p2vec[1])
end
function get_β(EB_on, pbar2)
    return 12 * EB_on / ((pbar2^2)*(3-pbar2))
end

∇W_poly(β, pbar2) = β*fromroots([0,1,pbar2])
W_poly(β, pbar2) = (β/12)*Polynomial([0, 6*pbar2, -3(pbar2+1), 2])

# Easy case - for zero input
function SDEdrift(emf::NoEF;
    v, EB_on, EB_off, D,
    kwargs...)

    β, pbar2 = _map_barriers_to_coefficients(EB_on, EB_off)
    ∇W = ∇W_poly(β, pbar2)
    
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
    v, EB_on, EB_off, D,
    γ1=0.0, γ2=0.0, γ3=0.0, γ4=0.0,
    kwargs...)

    β, pbar2 = _map_barriers_to_coefficients(EB_on, EB_off)
    ∇W = ∇W_poly(β, pbar2)

    dpol(pol,inp) = -D*(∇W(abs2(pol))*pol - γ4*inp)
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