export save, load

# SAVING
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
    write(g2, "log_sl", b.log_sl)
    write(g2, "ell", b.ell)
    
    close(fid)
end

function save(c::ConditionalExpectation, ::Type{ES}, fn::String="electro_data") where ES<:EmpiricalSummary
    
    fid = h5open(fn*".h5", "cw")
    
    g_name = String(Symbol(ES))
    overwrite_flag = has(fid, g_name)
 
    if overwrite_flag
        @info "Overwriting $ES data"
        g = fid[g_name]
        for dset_name in names(g)
            o_delete(g, dset_name)
        end
    else
        g = g_create(fid, g_name)
    end

    write(g, "D", c.D)
    write(g, "ell", c.ell)
    
    close(fid)
end


###### LOADING

function InferenceBatch(G::HDF5Group)
    str_names = read(attrs(G), "Names")
    Names = Tuple(map(Symbol, str_names))
    P = Parameters(read(G, "θ"), Names)
    ell = read(G, "ell")
    log_sl = read(G, "log_sl")
    return InferenceBatch(P, ell, log_sl)
end

function load(fn::String, ::Type{LT}, Names::NTuple{N, Symbol}) where {LT<:SyntheticLogLikelihood, N}
    fid = h5open(fn*".h5", "r")
    g1_name = String(Symbol(LT))
    g2_name = *(String.(Names)...)

    g1 = exists(fid, g1_name) ? fid[g1_name] : error("No data for any parameter space with synthetic likelihood type $LT")
    g2 = exists(g1, g2_name) ? g1[g2_name] : error("No data for parameter space $Names with synthetic likelihood type $LT")
    
    b = InferenceBatch(g2)
    close(fid)
    return b
end

function load(fn::String, ::Type{LT}) where LT<:SyntheticLogLikelihood
    fid = h5open(fn*".h5", "r")

    g_name = String(Symbol(LT))
    g = exists(fid, g_name) ? fid[g_name] : error("No data for any parameter space with synthetic likelihood type $LT")

    bvec = map(InferenceBatch, g)
    close(fid)
    return bvec
end

function ConditionalExpectation(G::HDF5Group)
    D = read(G, "D")
    ell = read(G, "ell")
    return ConditionalExpectation(D, ell)
end

function load(fn::String, ::Type{ES}) where ES<:EmpiricalSummary
    fid = h5open(fn*".h5", "r")
    g_name = String(Symbol(ES))
    
    g = exists(fid, g_name) ? fid[g_name] : error("No data for empirical summary type $ES")
    c = ConditionalExpectation(g)
    close(fid)
    return c
end

