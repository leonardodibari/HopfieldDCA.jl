struct HopPlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct HopPlmOut
    pslike::Union{Vector{Float64},Float64}
    Ktensor::Array{Float64,3}
    Vtensor::Array{Float64,2}
    score::Array{Tuple{Int, Int, Float64},1}  
end


struct HopPlmVar
    N::Int
    M::Float64
    q::Int    
    q2::Int
    H::Int
    lambdaK::Float64
    lambdaV::Float64
    Z::Array{Int,2}
    W::Array{Float64,1}
    delta_i::Array{Bool, 3}
    delta_j::Array{Bool, 3}
    delta_la::Array{Bool,2}
    
    function HopPlmVar(H, lambdaK, lambdaV, Z)
        println("creating alg ")
        W, M_eff = compute_weights(Z,0.2); 
        N = size(Z)[1]; q = maximum(Z);  q2=q*q;
        delta_la = Matrix(I,q,q) ; 
        @tullio delta_j[a, j, m] := a == Z[j, m] (a in 1:q); 
        @tullio delta_i[a, i, m] := a == Z[i, m] (a in 1:q);
        new(N,M_eff,q,q2,H,lambdaK, lambdaV, Z, W, delta_i, delta_j, delta_la)
    end
end


mutable struct StgArr
    #this are the ones for the likelyhood
    en::Array{Float64,3}
    data_en::Array{Float64,2}
    log_z::Array{Float64,2}
    loss::Array{Float64,1}
    reg_k::Array{Float64,1}
    
    v_prod::Array{Float64,4}
    prob::Array{Float64,3}
    grad_k1::Array{Float64,3}
    grad_k2::Array{Float64,3}
    grad_K::Array{Float64,3}
    tot_grad_K::Array{Float64,3}
    grad_v1::Array{Float64,3}
    grad_v2::Array{Float64,3}
    grad_V::Array{Float64,2}
    tot_grad_V::Array{Float64,2}
    

    function StgArr(plmvar)
        Z = plmvar.Z
        N = size(Z)[1]
        M = size(Z)[2] 
        H = plmvar.H
        q = plmvar.q
        en = zeros(q,N,M)
        data_en = zeros(N,M)
        log_z = zeros(N,M)
        loss = zeros(N)
        reg_k = zeros(N)

        v_prod = zeros(q,N,M,H)
        prob = zeros(q,N,M)
        grad_k1 = zeros(N,N,H)
        grad_k2 = zeros(N,N,H)
        grad_K = zeros(N,N,H)
        tot_grad_K = zeros(N,N,H)

        grad_v1 = zeros(N,M,H)
        grad_v2 = zeros(N,M,H)
        grad_V = zeros(N,H)
        tot_grad_V = zeros(N,H)

        new(en, data_en, log_z, loss, reg_k, v_prod, prob, 
        grad_k1, grad_k2, grad_K, tot_grad_K, grad_v1, grad_v2, grad_V, tot_grad_V)
    end
end

function define_var(filepath::String, H::Int, lambdaK::Float64, lambdaV::Float64, M::Int)

    Z = read_fasta_alignment(filepath,0)[:,1:M];

    alg_var = HopPlmVar(H, lambdaK, lambdaV, Z)
    
    return alg_var
end

function define_var(filepath::String, H::Int, lambdaK::Float64, lambdaV::Float64)
    Z = read_fasta_alignment(filepath,0);
    alg_var = HopPlmVar(H, lambdaK, lambdaV, Z)
    return alg_var
end   
