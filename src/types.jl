
struct HopPlmVar_gen{T1,T2,T3}
    N::Int
    q::Int    
    H::Int
    Z::Array{Int,2}
    K::T3
    V::T2
    W::T1
end

function HopPlmVar_gen(H, fastafile; T::DataType=Float32)
    println("new version")
    Z, W = quickread(fastafile)
    W = T.(W)
    N = size(Z,1); q = maximum(Z);  
    K = T.(rand(N, N, H)); V = T.(rand(q, H));
    potts_par = N*(N-1)*q*q/2;
    att_par = H*N^2  + q*H;
    ratio = round(att_par / potts_par, digits = 3);
    println("ratio=$ratio N=$N")
    T1 = typeof(W)
    T2 = typeof(V)
    T3 = typeof(K)
    println(T1,T2,T3)
    HopPlmVar_gen{T1,T2,T3}(N, q, H, Z, K, V, W)
end


struct HopPlmVar
    N::Int
    q::Int    
    H::Int
    Z::Array{Int,2}
    K::Array{Float64,3}
    V::Array{Float64,2}
    W::Array{Float64,1}
    ratio::Float64
end

function HopPlmVar(H, fastafile)
    println("new version")
    Z, W = quickread(fastafile)
    N = size(Z,1); q = maximum(Z);  
    K = rand(N, N, H); V = rand(q, H);
    potts_par = N*(N-1)*q*q/2;
    att_par = H*N^2  + q*H;
    ratio = round(att_par / potts_par, digits = 3);
    println("ratio=$ratio N=$N")
    HopPlmVar(N, q, H, Z, K, V, W, ratio)
end


mutable struct StgArr
    #this are the ones for the likelyhood
    KK::Array{Float64,3}
    J::Array{Float64,4}
    en::Array{Float64,3}
    data_en::Array{Float64,2}
    log_z::Array{Float64,2}
    loss::Array{Float64,1}
    delta_j::Array{Bool, 3}
end

function StgArr(plmvar)
    Z = plmvar.Z
    N = size(Z)[1]
    M = size(Z)[2] 
    H = plmvar.H
    q = plmvar.q
    
    KK = rand(N, N, H)
    J = rand(N,N,q,q)
    en = zeros(q,N,M)
    data_en = zeros(N,M)
    log_z = zeros(N,M)
    loss = zeros(N)
    @tullio delta_j[a, j, m] := a == Z[j, m] (a in 1:q); 
    
    StgArr(KK, J, en, data_en, log_z, loss, delta_j)
end


mutable struct Stg{T1, T2, T3, T4}
    KK::T3
    J::T4
    en::T3
    v_prod::T4
    log_z::T2
    prob::T3
    data_en::T2
    loss::T1
    grad_k11::T4
    grad_k1::T3
    grad_k2::T3
    grad_k::T3
    gg_A::T3
    gg_BB::T3
    gg_B2::T4
    gg_B::T3
    grad_v1::T3
    gg_C::T4
    grad_v2::T3
    grad_V::T2
    tot_grad_K::T3
    reg_v01::T4
    reg_v02::T4
    reg_v1::T3
    reg_v2::T3
    reg_v::T2
    tot_grad_V::T2
end

function Stg(plmvar; m = 0)
    Z = plmvar.Z
    N = size(Z)[1]
    if m == 0
        M = size(Z)[2]
    else
        M = m
    end

    H = plmvar.H
    q = plmvar.q

    T1 = typeof(plmvar.W)
    T2 = typeof(plmvar.V)
    T3 = typeof(plmvar.K)
    T = eltype(plmvar.K)
    T4 = Array{T,4}

    KK = rand(N, N, H)
    J = rand(N,N,q,q)
    en = zeros(q,N,M)
    v_prod = zeros(q,N,M,H)
    log_z = zeros(N,M)
    prob = zeros(q,N,M)
    data_en = zeros(N,M)
    loss = zeros(N)
    grad_k11 = zeros(N,N,M,H)
    grad_k1 = zeros(N,N,H)
    grad_k2 = zeros(N,N,H)
    grad_k = zeros(N,N,H)
    gg_A = zeros(q,M,H)
    gg_BB = zeros(N,M,H)
    gg_B2 = zeros(N,q,M,H)
    gg_B = zeros(q,M,H)
    grad_v1 = zeros(q,M,H)
    gg_C = zeros(N,q,M,H)
    grad_v2 = zeros(q,M,H)
    grad_V = zeros(q,H)
    tot_grad_K = zeros(N,N,H)
    reg_v01 = zeros(N,N,q,H)
    reg_v02 = zeros(N,N,q,H)
    reg_v1 = zeros(N,q,H)
    reg_v2 = zeros(N,q,H)
    reg_v = zeros(q,H)
    tot_grad_V = zeros(q,H) 
    
    Stg{T1, T2, T3, T4}(KK, J, en, v_prod, log_z, prob, data_en, loss, grad_k11, grad_k1, grad_k2, grad_k, gg_A, 
    gg_BB, gg_B2, gg_B, grad_v1, gg_C, grad_v2, grad_V, tot_grad_K, reg_v01, reg_v02, reg_v1, reg_v2, reg_v, 
    tot_grad_V)

end
