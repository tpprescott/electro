export save, load

function save(b::InferenceBatch{Names}, ::Type{LT}, fn::String="electro_data") where {Names, LT<:SyntheticLogLikelihood}
    
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(Symbol(LT))
    g2_name = *(String.(Names)...)

    g1 = exists(fid, g1_name) ? fid[g1_name] : g_create(fid, g1_name)

    overwrite_flag = has(g1, g2_name)
    if overwrite_flag
        @info "Overwriting $Names data for $LT"
        g2 = g1[g2_name]
        for dset_name in names(g2)
            o_delete(g2, dset_name)
        end
    else
        g2 = g_create(g1, g2_name)
        attrs(g2)["Names"] = [String(nm) for nm in Names]
    end

    write(g2, "θ", b.θ.θ)
    write(g2, "log_π", b.log_π)
    write(g2, "log_q", b.log_q)
    write(g2, "log_sl", b.log_sl)
    write(g2, "log_u", b.log_u)
    
    close(fid)
end

function InferenceBatch(G::HDF5Group)
    str_names = read(attrs(G), "Names")
    Names = Tuple(map(Symbol, str_names))
    P = Parameters(read(G, "θ"), Names)
    log_π = read(G, "log_π")
    log_q = read(G, "log_q")
    log_sl = read(G, "log_sl")
    log_u = read(G, "log_u")
    return InferenceBatch(P, log_π, log_q, log_sl, log_u)
end

function load(fn::String, ::Type{LT}, Names::NTuple{N, Symbol}) where {LT<:SyntheticLogLikelihood, N}
    fid = h5open(fn*".h5", "r")
    g1_name = String(Symbol(LT))
    g2_name = *(String.(Names)...)

    g1 = exists(fid, g1_name) ? fid[g1_name] : error("No data for any parameter space with synthetic likeilhood type $LT")
    g2 = exists(g1, g2_name) ? g1[g2_name] : error("No data for parameter space $Names with synthetic likeilhood type $LT")
    
    b = InferenceBatch(g2)
    close(fid)
    return b
end

function load(fn::String, ::Type{LT}) where LT<:SyntheticLogLikelihood
    fid = h5open(fn*".h5", "r")

    g_name = String(Symbol(LT))
    g = exists(fid, g_name) ? fid[g_name] : error("No data for any parameter space with synthetic likeilhood type $LT")

    bvec = map(InferenceBatch, g)
    close(fid)
    return bvec
end