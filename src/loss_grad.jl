
function wrapper(x::Vector, g::Vector, plmvar::HopPlmVar)

    q = plmvar.q
    N = plmvar.N
    M = plmvar.M
    Z = plmvar.Z
    H = plmvar.H
    W = plmvar.W
    
    K = x[1:N*N*H]
    V = x[N*N*H+1:end]
    g_K = g[1:N*N*H]
    g_V = g[N*N*H+1:end]
    return get_pl_and_grad(K, V, g_K, g_V, plm_var)
end

function get_pl_and_grad(K::Array{Float64,3}, V::Array{Float64,2}, g_K::Vector{Float64}, g_V::Vector{Float64},
     plmvar::HopPlmVar)
   
    q = plmvar.q
    N = plmvar.N
    M = plmvar.M
    Z = plmvar.Z
    H = plmvar.H
    W = plmvar.W
    lambdaK = plmvar.lambdaK
    lambdaV = plmvar.lambdaV

    grad_K = reshape(g_K, (N, N, H))
    grad_V = reshape(g_V, (q, H))

    # N = 30; i = 1; ; M = 50; H = 10; K = rand(N,N,H); V = rand(20,H); Z = rand(1:20, N, M); W = rand(M)
    #order for optimal access is a, i/j, h, m
    # deltas should be stored once for all, maybe in the struct
    @tullio delta_i[a, i, m] := a .==  a .== Z[i, m] (a in 1:20)
    @tullio delta_j[a, j, m] := a .==  a .== Z[j, m] (a in 1:20)
    

    #useful quantities
    @tullio v_prod[a, j, h, m] := V[a, h]*V[Z[j, m], h]
    @tullio en[a, i, m] := K[i, j, h]*v_prod[a, j, h, m]* (j != i)
    @tullio no_norm_prob[a, i, m] := exp(en[a, i, m])
    @tullio z[i, m] := no_norm_prob[a, i, m]
    log_z = log.(z)
    @tullio data_en[i, m] := en[Z[i, m], i, m]

    #parts of the gradient in lambdaK 
    @tullio grad_k1[i, j, h, m] := no_norm_prob[a, i, m]*v_prod[a, j, h, m]/z[i,m]* (j != i)
    @tullio grad_k2[i, j, h, m] := v_prod[Z[i, m], j, h, m]

    #parts of the gradient in V  
    @tullio grad_v1[a, h, m] := no_norm_prob[a, i, m]*K[i, j, h]*V[Z[j, m], h]*(1+delta_j[a, j, m])/z[i,m]* (j != i)
    @tullio grad_v2[h, m] := K[i, j, h]*(V[Z[j, m], h]*delta_i[a, i, m] + V[Z[i, m], h]*delta_j[a, j, m])

    #likelihood
    @tullio like[i] := W[m]*(log_z[i, m] - data_en[i,m])

    #gradients
    @tullio grad_K[i, j, h] = W[m]*(grad_k1[i, j, h, m] - grad_k2[i, j, h, m])
    @tullio grad_V[a, h] = W[m]*(grad_v1[a, h, m] - grad_v2[h, m])


    #regularization
    @tullio like[i] = like[i] + lambdaK*K[i, j, h]*K[i, j, h] + lambdaV*V[a, h]*V[a, h] * (j != i)

    g_K = reshape(grad_K, prod(size(grad_K)))
    g_V = reshape(grad_K, prod(size(grad_V)))
     
    return like 
end


"""NLopt tutotial
using NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 2*x[2]
    end
    return x[2]^2
end



opt = Opt(:LD_MMA, 2)
opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc

(minf,minx,ret) = optimize(opt, [1.234, 5.4])
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")

"""