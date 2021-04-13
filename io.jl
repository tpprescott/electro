export save, load, asave, aload

# SAVING
function save(b::InferenceBatch{P}, LT::Symbol; fn::String="electro_data") where P<:Parameters{Names,1} where Names
    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(LT)
    g2_name = *(String.(Names)...)

    g1 = haskey(fid, g1_name) ? fid[g1_name] : create_group(fid, g1_name)

    overwrite_flag = haskey(g1, g2_name)
    if overwrite_flag
        @info "Overwriting $Names data for $LT"
        g2 = g1[g2_name]
        for dset in g2
            delete_object(dset)
        end
    else
        g2 = create_group(g1, g2_name)
        attributes(g2)["Names"] = [String(nm) for nm in Names]
    end

    parvec(p::Particle) = p.θ.θ
    θmat = hcat(parvec.(b)...)
    write(g2, "θ", θmat)
    write(g2, "log_sl", b.log_sl)
    write(g2, "ell", b.ell)
    
    close(fid)
end

function save(c::ConditionalExpectation, ES::Symbol; fn::String="electro_data")
    eval(ES) <: EmpiricalSummary || error("Wrong symbol, mush --- $(eval(ES)) is not subtype of EmpiricalSummary")
    
    fid = h5open(fn*".h5", "cw")
    
    g_name = String(ES)
    overwrite_flag = haskey(fid, g_name)
 
    if overwrite_flag
        @info "Overwriting $ES conditional expectations"
        g = fid[g_name]
        for dset in g
            delete_object(dset)
        end
    else
        g = create_group(fid, g_name)
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
    
    g1 = haskey(fid, g1_name) ? fid[g1_name] : create_group(fid, g1_name)
    g2 = haskey(g1, g2_name) ? g1[g2_name] : create_group(g1, g2_name)
    
    overwrite_flag = haskey(g2, String(attr_name))
    if overwrite_flag
        @info "Overwriting $attr_name attribute for $LT, $Names data"
        delete_attribute(g2, String(attr_name))
    end
    attributes(g2)[String(attr_name)] = a
    close(fid)
end


###### LOADING INFERENCE BATCHES

function InferenceBatch(G::HDF5.Group)
    str_names = read(attributes(G), "Names")
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

    g1 = fid[g1_name]
    g2 = g1[g2_name]
    
    b = InferenceBatch(g2)
    close(fid)
    return b
end


########## LOAD CONDITIONAL EXPECTATIONS

function ConditionalExpectation(G::HDF5.Group, Names)
    D = read(G, "D")
    ell = read(G, "ell")
    return ConditionalExpectation(D, ell, Names)
end

function load(ES::Symbol; fn::String="electro_data")
    ES_type = eval(ES)
    ES_type <: EmpiricalSummary || error("Wrong symbol, mush --- $(eval(ES)) is not subtype of EmpiricalSummary")
    fid = h5open(fn*".h5", "r")
    g_name = String(ES)

    g = fid[g_name]
    
    c = ConditionalExpectation(g, getnames(eval(ES)))
    close(fid)
    return c
end

function aload(attr_name, LT::Symbol, Names::NTuple{N, Symbol}; fn::String="electro_data") where N
    eval(LT) <: SyntheticLogLikelihood || error("Wrong symbol, mush --- $(eval(LT)) is not subtype of SyntheticLogLikelihood")
    fid = h5open(fn*".h5", "cw")
    
    g1_name = String(LT)
    g2_name = *(String.(Names)...)
    
    g1 = fid[g1_name]
    g2 = g1[g2_name]

    a = read(attributes(g2), String(attr_name))
    close(fid)
    return a
end