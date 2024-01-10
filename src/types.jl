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
    M::Int
    q::Int    
    q2::Int
    H::Int
    lambdaK::Float64
    lambdaV::Float64
    Z::Array{Int,2}
    W::Array{Float64,1}
    
    function HopPlmVar(N,M,q,H,lambdaK, lambdaV, Z,W)
        q2=q*q
        new(N,M,q,q2,H,lambdaK, lambdaV, Z, W)
    end
end