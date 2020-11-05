export save, load

# SAVING
function save(b::InferenceBatch{Names}, LT::Symbol; fn::String="electro_data") where Names
    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(LT)
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

function save(c::ConditionalExpectation, ES::Symbol; fn::String="electro_data")
    eval(ES) <: EmpiricalSummary || error("Wrong symbol, mush --- $(eval(ES)) is not subtype of EmpiricalSummary")
    
    fid = h5open(fn*".h5", "cw")
    
    g_name = String(ES)
    overwrite_flag = has(fid, g_name)
 
    if overwrite_flag
        @info "Overwriting $ES conditional expectations"
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

function asave(a, attr_name, LT::Symbol, Names::NTuple{N, Symbol}; fn::String="electro_data") where N

    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(LT)
    g2_name = *(String.(Names)...)
    
    g1 = exists(fid, g1_name) ? fid[g1_name] : g_create(fid, g1_name)
    g2 = exists(g1, g2_name) ? g1[g2_name] : g_create(g1, g2_name)
    
    overwrite_flag = has(g2, String(attr_name))
    if overwrite_flag
        @info "Overwriting $attr_name attribute for $LT, $Names data"
        a_delete(g2, String(attr_name))
    end
    attrs(g2)[String(attr_name)] = a
    close(fid)
end


###### LOADING INFERENCE BATCHES

function InferenceBatch(G::HDF5Group)
    str_names = read(attrs(G), "Names")
    Names = Tuple(map(Symbol, str_names))
    P = Parameters(read(G, "θ"), Names)
    ell = read(G, "ell")
    log_sl = read(G, "log_sl")
    return InferenceBatch(P, ell, log_sl)
end

function load(LT::Symbol, Names::NTuple{N, Symbol}; fn::String="electro_data") where N
    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    fid = h5open(fn*".h5", "r")
    g1_name = String(LT)
    g2_name = *(String.(Names)...)

    g1 = exists(fid, g1_name) ? fid[g1_name] : error("No data for any parameter space with synthetic likelihood type $LT")
    g2 = exists(g1, g2_name) ? g1[g2_name] : error("No data for parameter space $Names with synthetic likelihood type $LT")
    
    b = InferenceBatch(g2)
    close(fid)
    return b
end


########## LOAD CONDITIONAL EXPECTATIONS

function ConditionalExpectation(G::HDF5Group, Names)
    D = read(G, "D")
    ell = read(G, "ell")
    return ConditionalExpectation(D, ell, Names)
end

function load(ES::Symbol; fn::String="electro_data")
    eval(ES) <: EmpiricalSummary || error("Wrong symbol, mush --- $(eval(ES)) is not subtype of EmpiricalSummary")
    fid = h5open(fn*".h5", "r")
    g_name = String(ES)

    g = exists(fid, g_name) ? fid[g_name] : error("No data for summary type $ES")
    
    c = ConditionalExpectation(g2, getnames(eval(ES)))
    close(fid)
    return c
end

function aload(attr_name, LT::Symbol, Names::NTuple{N, Symbol}; fn::String="electro_data") where N
    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(LT)
    g2_name = *(String.(Names)...)
    
    g1 = exists(fid, g1_name) ? fid[g1_name] : error("No $LT data")
    g2 = exists(g1, g2_name) ? g1[g2_name] : error("No $Names data for $LT")

    has(attrs(g2), String(attr_name)) || error("No $attr_name attribute for $LT and $Names")
    a = read(attrs(g2), String(attr_name))
    close(fid)
    return a
end