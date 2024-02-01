

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
    
    StgArr(KK, J, en, data_en, log_z, loss)
end
