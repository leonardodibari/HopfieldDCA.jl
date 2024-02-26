struct HopPlmVar_full{T1,T2} 
    N::Int
    q::Int    
    H::Int
    Z::Array{Int,2}
    K::T2
    V::T2
    W::T1
end

function HopPlmVar_full(H, fastafile; T::DataType=Float32)
    println("final version")
    Z, W = quickread(fastafile)
    W = T.(W)
    N = size(Z,1); q = maximum(Z);  
    K = T.(rand(N, H)); V = T.(rand(q, H));
    potts_par = N*(N-1)*q*q/2;
    att_par = H*N^2  + q*H;
    ratio = round(att_par / potts_par, digits = 3);
    println("ratio=$ratio N=$N")
    T1 = typeof(W)
    T2 = typeof(V)
    HopPlmVar_full{T1,T2}(N, q, H, Z, K, V, W)
end





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
    println("final version")
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
    HopPlmVar_gen{T1,T2,T3}(N, q, H, Z, K, V, W)
end


struct Stg{T1, T2, T3, T4}
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

function Stg(plmvar::HopPlmVar_gen; m = 0)
    Z = plmvar.Z
    N = size(Z,1)
    if m == 0
        M = size(Z,2)
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




struct SmallStg{T1, T2, T3, T4}
    KK::T3
    J::T4
    en::T2
    v_prod::T3
    log_z::T1
    prob::T2
    data_en::T1
    loss::T1
    grad_k11::T3
    grad_k1::T3
    grad_k2::T3
    grad_k::T3
    gg_A::T2
    gg_BB::T2
    gg_B2::T3
    gg_B::T2
    grad_v1::T2
    gg_C::T3
    grad_v2::T2
    grad_V::T2
    tot_grad_K::T3
    reg_v01::T4
    reg_v02::T4
    reg_v1::T3
    reg_v2::T3
    reg_v::T2
    tot_grad_V::T2
end



function SmallStg(plmvar::HopPlmVar_gen)
    Z = plmvar.Z
    N = size(Z,1)
    H = plmvar.H
    q = plmvar.q

    T1 = typeof(plmvar.W)
    T2 = typeof(plmvar.V)
    T3 = typeof(plmvar.K)
    T = eltype(plmvar.K)
    T4 = Array{T,4}
    KK = rand(N, N, H)
    J = rand(N,N,q,q)
    en = zeros(q,N)  # Removed M dimension
    v_prod = zeros(q,N,H)  # Removed M dimension
    log_z = zeros(N)  # Removed M dimension
    prob = zeros(q,N)  # Removed M dimension
    data_en = zeros(N)  # Removed M dimension
    loss = zeros(N)
    grad_k11 = zeros(N,N,H)  # Removed M dimension
    grad_k1 = zeros(N,N,H)
    grad_k2 = zeros(N,N,H)
    grad_k = zeros(N,N,H)
    gg_A = zeros(q,H)  # Removed M dimension
    gg_BB = zeros(N,H)  # Removed M dimension
    gg_B2 = zeros(N,q,H) #Removed M dimension
    gg_B = zeros(q,H)  # Removed M dimension
    grad_v1 = zeros(q,H)  # Removed M dimension
    gg_C = zeros(N,q,H)  # Removed M dimension
    grad_v2 = zeros(q,H)  # Removed M dimension
    grad_V = zeros(q,H)
    tot_grad_K = zeros(N,N,H)
    reg_v01 = zeros(N,N,q,H)
    reg_v02 = zeros(N,N,q,H)
    reg_v1 = zeros(N,q,H)
    reg_v2 = zeros(N,q,H)
    reg_v = zeros(q,H)
    tot_grad_V = zeros(q,H) 
    
    SmallStg{T1, T2, T3, T4}(KK, J, en, v_prod, log_z, prob, data_en, loss, grad_k11, grad_k1, grad_k2, grad_k, gg_A, 
    gg_BB, gg_B2, gg_B, grad_v1, gg_C, grad_v2, grad_V, tot_grad_K, reg_v01, reg_v02, reg_v1, reg_v2, reg_v, 
    tot_grad_V)
end


