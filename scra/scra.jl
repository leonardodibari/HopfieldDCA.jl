"""
function get_loss_pagnani(K::Array{Float64,3},
    V::Array{Float64,2},
    plmvar::HopPlmVar;lambdaK = 0.005, lambdaV = 0.005)

    Z = plmvar.Z
    W = plmvar.W
    M = size(Z,2)
   

    #useful quantities
    W .= W/sum(W)

    en = zeros(plmvar.q, plmvar.N,M)
    data_en = zeros(plmvar.N, M)
    log_z = zeros(plmvar.N, M)
    loss = zeros(plmvar.N)
    reg_v = 0.0
    
    @inbounds @simd for m in 1:M
        for i in axes(K,1)
            for a in axes(en,1)
                for j in axes(K,2)
                    if j != i
                        for h in axes(K,3)
                            en[a, i, m] += K[i, j, h]*V[a, h]*V[Z[j, m], h]
                        end
                    end
                end
            end
        end
    end

    log_z = logsumexp(en)[1, :, :]
    @inbounds @fastmath for i in 1:plmvar.N
        for m in 1:M
            data_en[i, m] = en[Z[i, m], i, m]
            loss[i] += W[m] * (log_z[i, m] - data_en[i, m])
        end
    end

    reg_v = lambdaV*sum(abs2, V)
    sreg_k = 0.0
    @inbounds @fastmath for i in axes(K,1)
        for j in axes(K,2)
            if j != i
                for h in 1:plmvar.H
                    sreg_k += K[i,j,h] * K[i,j,h]
                end
            end
        end
    end
    sreg_k *= lambdaK

    #println("loss is $(sum(loss)) + $(sreg_k) + $reg_v and sumn=$(sum(en)) sumn=$(sum(data_en))")
    return sum(loss)+ sreg_k + reg_v
end

function get_loss_zyg_francesco(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    plmvar::HopPlmVar;lambdaK = 0.005, lambdaV = 0.005)
  
    Z = plmvar.Z
    W = plmvar.W
    
    Wt = W/sum(W)
    

    @tullio J[i,j,a,b] := K[i,j,h]*V[a,h]*V[b,h]*(j!=i)
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := Wt[m]*(log_z[i, m] - data_en[i,m])
    
    #regularization
    @tullio reg_v := lambdaV*V[a, h]*V[a, h]
    @tullio sreg_k := lambdaK * K[i,j,h] * K[i,j,h] * (j!=i)
    #println("loss is $(sum(loss)) + $(sreg_k) + $reg_v and sumn=$(sum(en)) and sumn=$(sum(data_en))")
    return sum(loss) + sreg_k + reg_v 
end

function get_loss_zyg(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    plmvar::HopPlmVar;lambdaK = 0.005, lambdaV = 0.005)
  
    Z = plmvar.Z
    W = plmvar.W
    Wt = W/sum(W)
    

    @tullio en[a, i, m] :=  K[i, j, h] * V[a, h] * V[Z[j, m], h] * (j != i)
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := Wt[m]*(log_z[i, m] - data_en[i,m])
    
    #regularization
    @tullio reg_v := lambdaV*V[a, h]*V[a, h]
    @tullio sreg_k := lambdaK * K[i,j,h] * K[i,j,h] * (j!=i)
    #println("loss is $(sum(loss)) + $(sreg_k) + $reg_v and sumn=$(sum(en)) and sumn=$(sum(data_en))")
    return sum(loss) + sreg_k + reg_v 
end


function get_loss_and_grad(K::Array{Float64,3}, 
    V::Array{Float64,2},
    Z::Array{Int,2}, 
    _w::Array{Float64, 1},
    delta_j; lambda = 0.001)
   

    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)

    
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio v_prod[a, j, m, h] := V[a, h]*V[Z[j, m], h]
    log_z = logsumexp(en)[1,:,:]
    
    @tullio prob[a,i,m] := exp(en[a,i,m] - log_z[i,m]) 
    @tullio grad_k11[i, j, m, h] := prob[a, i, m]*v_prod[a, j, m, h] * (j!=i) 
    @tullio grad_k1[i, j, h] := _w[m] * grad_k11[i, j, m, h]
    @tullio grad_k2[i, j, h] := _w[m] * v_prod[Z[i, m], j, m, h] * (j!=i) 
    grad_K = grad_k1 .- grad_k2

    @tullio tot_grad_K[i, j, h] := grad_K[i, j, h] + 2*lambda*J[i, j, a, b]*V[a, h]*V[b, h] 
    
    @tullio gg_A[l, m, h] := prob[l, i, m]*V[Z[j, m], h]*KK[i, j, h]
    @tullio gg_BB[i, m, h] := prob[a, i, m]*V[a,h]
    @tullio gg_B2[i,l,m,h] := gg_BB[i, m, h]*delta_j[l, j, m]*KK[i, j, h]
    @tullio gg_B[l,m,h] := gg_B2[i,l,m,h]
    @tullio grad_v1[l, m, h] := gg_A[l, m, h] + gg_B[l, m, h]

    @tullio gg_C[i,l,m,h] := KK[i, j, h]*(V[Z[i, m], h]*delta_j[l, j, m]+V[Z[j, m], h]*delta_j[l, i, m])
    @tullio grad_v2[l, m, h] := gg_C[i,l,m,h]    
    @tullio grad_V[l, h] := _w[m]*(grad_v1[l, m, h] - grad_v2[l, m, h])
    
    @tullio reg_v01[i,j,l,h] := J[i,j,a,l]*V[a,h]
    @tullio reg_v02[i,j,l,h] := J[i,j,l,b]*V[b,h]
    @tullio reg_v1[i,l,h] := K[i, j, h]*reg_v01[i,j,l,h]
    @tullio reg_v2[i,l,h] := K[i, j, h]*reg_v02[i,j,l,h]
    @tullio reg_v[l, h] := reg_v1[i, l, h] + reg_v2[i, l, h]
    
    @tullio tot_grad_V[l, h] := grad_V[l, h] + 2*lambda*reg_v[l,h]

    return  tot_grad_K, tot_grad_V
end
"""
