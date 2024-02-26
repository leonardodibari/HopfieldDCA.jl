function get_loss_J(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001), ort = 50) where {T}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
  
    return sum(loss) + lambda * sum(abs2, J)
end

function get_loss_J_orthog(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001), ort = 50) where {T}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
  
    ort_loss = orthotullio(V)
    
    return sum(loss) + lambda * sum(abs2, J) + ort * ort_loss
end

function get_loss_J_parts(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001), ort = 50) where {T}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    return round(sum(loss), digits = 3), round(lambda * sum(abs2, J), digits = 3) 
end


function get_loss_J_orthog_parts(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001), ort = 50) where {T}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    ort_loss = orthotullio(V)
    
    return round(sum(loss), digits = 3), round(lambda * sum(abs2, J) + ort * ort_loss, digits = 3)
end

function get_loss_fullJ(K::Array{T,2}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001)) where {T}
    
    @tullio KK[i,j,h] := K[i,h]*K[j,h]
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
  
    return sum(loss) + lambda * sum(abs2, J)
end

function get_loss_fullJ_parts(K::Array{T,2}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda::T = T(0.001)) where {T}
    
    @tullio KK[i,j,h] := K[i,h]*K[j,h]
    @tullio J[a,i,b,j] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[a, i, Z[j, m], j]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    #log_z = logsumexp(en)[1,:,:]
    log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    return round(sum(loss), digits = 3), round(lambda * sum(abs2, J), digits = 3) 
end


function get_loss_parts(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}, tmp::Stg; lambda = 0.001) where {T<:AbstractFloat}
   
    @tullio tmp.KK[i,j,h] = K[i,j,h]*(j!=i)

    @tullio tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    @tullio tmp.en[a, i, m] = tmp.J[i, j, a, Z[j, m]]
    @tullio tmp.data_en[i, m] = tmp.en[Z[i, m], i, m]
    tmp.log_z = logsumexp(tmp.en)[1,:,:]
    @tullio tmp.loss[i] := _w[m]*(tmp.log_z[i, m] - tmp.data_en[i,m])

    return round(sum(tmp.loss), digits = 3), round(lambda * sum(abs2, tmp.J), digits = 3)
end

function single_site!(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T,1}, 
    tmp::SmallStg, 
    m::Int) where {T<:AbstractFloat}

    @tullio grad=false tmp.en[a, i] = tmp.J[i, j, a, Z[j, m]]
    @tullio grad=false tmp.v_prod[a, j, h] = V[a, h]*V[Z[j, m], h]
    tmp.log_z .= logsumexp(tmp.en)[1,:]

    @tullio grad=false tmp.prob[a,i] = exp(tmp.en[a,i] - tmp.log_z[i]) 
    @tullio grad=false tmp.grad_k11[i, j, h] = tmp.prob[a, i]*tmp.v_prod[a, j, h] * (j!=i) 
    @tullio grad=false tmp.grad_k1[i, j, h] = _w[m] * tmp.grad_k11[i, j, h]
    @tullio grad=false tmp.grad_k2[i, j, h] = _w[m] * tmp.v_prod[Z[i, m], j, h] * (j!=i) 
    @tullio grad=false tmp.grad_k[i, j, h] = tmp.grad_k[i, j, h] + tmp.grad_k1[i, j, h] - tmp.grad_k2[i, j, h]

    @tullio grad=false tmp.gg_A[l, h] = tmp.prob[l, i]*V[Z[j, m], h]*tmp.KK[i, j, h]
    @tullio grad=false tmp.gg_BB[i, h] = tmp.prob[a, i]*V[a,h]
    @tullio grad=false tmp.gg_B2[i, l, h] = tmp.gg_BB[i, h]*(l==Z[j, m])*tmp.KK[i, j, h]
    @tullio grad=false tmp.gg_B[l, h] = tmp.gg_B2[i, l, h]
    @tullio grad=false tmp.grad_v1[l, h] = tmp.gg_A[l, h] + tmp.gg_B[l, h]

    @tullio grad=false tmp.gg_C[i,l,h] = tmp.KK[i, j, h]*(V[Z[i, m], h]*(l== Z[j, m])+ V[Z[j, m], h]*(l==Z[i, m]))
    @tullio grad=false tmp.grad_v2[l, h] = tmp.gg_C[i,l,h]    
    @tullio grad=false tmp.grad_V[l, h] = tmp.grad_V[l, h] + _w[m]*(tmp.grad_v1[l, h] - tmp.grad_v2[l, h])

end


function get_new_grad(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T,1}, 
    tmp::SmallStg,
    n_m::Int; 
    lambda = 0.001) where {T<:AbstractFloat}
   
    println("new version")
    
    @tullio grad=false tmp.KK[i,j,h] = K[i,j,h]*(j!=i)
    @tullio grad=false tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    
    fill!(tmp.grad_k, 0)
    fill!(tmp.grad_V, 0)

    @fastmath @inbounds for m in 1:n_m #length(_w)
        single_site!(K, V, Z, _w, tmp, m)         
    end

    @tullio grad=false  tmp.tot_grad_K[i, j, h] = tmp.grad_k[i, j, h] + 2*lambda*tmp.J[i, j, a, b]*V[a, h]*V[b, h] 
    
    @tullio grad=false  tmp.reg_v01[i,j,l,h] = tmp.J[i,j,a,l]*V[a,h]
    @tullio grad=false  tmp.reg_v02[i,j,l,h] = tmp.J[i,j,l,b]*V[b,h]
    @tullio grad=false  tmp.reg_v1[i,l,h] = K[i, j, h]*tmp.reg_v01[i,j,l,h]
    @tullio grad=false  tmp.reg_v2[i,l,h] = K[i, j, h]*tmp.reg_v02[i,j,l,h]
    @tullio grad=false  tmp.reg_v[l, h] = tmp.reg_v1[i, l, h] + tmp.reg_v2[i, l, h]
    
    @tullio grad=false  tmp.tot_grad_V[l, h] = tmp.grad_V[l, h] + 2*lambda*tmp.reg_v[l,h]
    
    return  (dK=tmp.tot_grad_K, dV=tmp.tot_grad_V)
end

function get_anal_grad(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}, tmp::Stg; lambda = 0.001) where {T<:AbstractFloat}
   
    @tullio grad=false tmp.KK[i,j,h] = K[i,j,h]*(j!=i)

    @tullio grad=false tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    @tullio grad=false tmp.en[a, i, m] = tmp.J[i, j, a, Z[j, m]]
    @tullio grad=false  tmp.v_prod[a, j, m, h] = V[a, h]*V[Z[j, m], h]
    tmp.log_z .= logsumexp(tmp.en)[1,:,:]

    @tullio grad=false  tmp.prob[a,i,m] = exp(tmp.en[a,i,m] - tmp.log_z[i,m]) 
    @tullio grad=false  tmp.grad_k11[i, j, m, h] = tmp.prob[a, i, m]*tmp.v_prod[a, j, m, h] * (j!=i) 
    @tullio grad=false  tmp.grad_k1[i, j, h] = _w[m] * tmp.grad_k11[i, j, m, h]
    @tullio grad=false  tmp.grad_k2[i, j, h] = _w[m] * tmp.v_prod[Z[i, m], j, m, h] * (j!=i) 
    tmp.grad_k .= tmp.grad_k1 .- tmp.grad_k2

    @tullio grad=false  tmp.tot_grad_K[i, j, h] = tmp.grad_k[i, j, h] + 2*lambda*tmp.J[i, j, a, b]*V[a, h]*V[b, h] 
    
    @tullio grad=false  tmp.gg_A[l, m, h] = tmp.prob[l, i, m]*V[Z[j, m], h]*tmp.KK[i, j, h]
    @tullio grad=false  tmp.gg_BB[i, m, h] = tmp.prob[a, i, m]*V[a,h]
    @tullio grad=false  tmp.gg_B2[i,l,m,h] = tmp.gg_BB[i, m, h]*(l==Z[j, m])*tmp.KK[i, j, h]
    @tullio grad=false  tmp.gg_B[l,m,h] = tmp.gg_B2[i,l,m,h]
    @tullio grad=false  tmp.grad_v1[l, m, h] = tmp.gg_A[l, m, h] + tmp.gg_B[l, m, h]

    @tullio grad=false  tmp.gg_C[i,l,m,h] = tmp.KK[i, j, h]*(V[Z[i, m], h]*(l== Z[j, m])+V[Z[j, m], h]*(l==Z[i, m]))
    @tullio grad=false  tmp.grad_v2[l, m, h] = tmp.gg_C[i,l,m,h]    
    @tullio grad=false  tmp.grad_V[l, h] = _w[m]*(tmp.grad_v1[l, m, h] - tmp.grad_v2[l, m, h])
    
    @tullio grad=false  tmp.reg_v01[i,j,l,h] = tmp.J[i,j,a,l]*V[a,h]
    @tullio grad=false  tmp.reg_v02[i,j,l,h] = tmp.J[i,j,l,b]*V[b,h]
    @tullio grad=false  tmp.reg_v1[i,l,h] = K[i, j, h]*tmp.reg_v01[i,j,l,h]
    @tullio grad=false  tmp.reg_v2[i,l,h] = K[i, j, h]*tmp.reg_v02[i,j,l,h]
    @tullio grad=false  tmp.reg_v[l, h] = tmp.reg_v1[i, l, h] + tmp.reg_v2[i, l, h]
    
    @tullio grad=false  tmp.tot_grad_V[l, h] = tmp.grad_V[l, h] + 2*lambda*tmp.reg_v[l,h]

    return  (dK=tmp.tot_grad_K, dV=tmp.tot_grad_V)
end

