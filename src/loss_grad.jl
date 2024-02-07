function get_loss_J(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda = 0.001) where {T<:AbstractFloat}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
  
    return sum(loss) + lambda * sum(abs2, J)
end

function get_loss_J_parts(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}; lambda = 0.001) where {T<:AbstractFloat}
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
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

function get_anal_grad(K::Array{T,3}, 
    V::Array{T,2}, 
    Z::Array{Int,2}, 
    _w::Array{T, 1}, tmp::Stg, delta_j; lambda = 0.001) where {T<:AbstractFloat}
   
    @tullio tmp.KK[i,j,h] = K[i,j,h]*(j!=i)

    @tullio tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    @tullio tmp.en[a, i, m] = tmp.J[i, j, a, Z[j, m]]
    @tullio tmp.v_prod[a, j, m, h] = V[a, h]*V[Z[j, m], h]
    tmp.log_z = logsumexp(tmp.en)[1,:,:]

    @tullio tmp.prob[a,i,m] = exp(tmp.en[a,i,m] - tmp.log_z[i,m]) 
    @tullio tmp.grad_k11[i, j, m, h] = tmp.prob[a, i, m]*tmp.v_prod[a, j, m, h] * (j!=i) 
    @tullio tmp.grad_k1[i, j, h] = _w[m] * tmp.grad_k11[i, j, m, h]
    @tullio tmp.grad_k2[i, j, h] = _w[m] * tmp.v_prod[Z[i, m], j, m, h] * (j!=i) 
    tmp.grad_k = tmp.grad_k1 .- tmp.grad_k2

    @tullio tmp.tot_grad_K[i, j, h] = tmp.grad_k[i, j, h] + 2*lambda*tmp.J[i, j, a, b]*V[a, h]*V[b, h] 
    
    @tullio tmp.gg_A[l, m, h] = tmp.prob[l, i, m]*V[Z[j, m], h]*tmp.KK[i, j, h]
    @tullio tmp.gg_BB[i, m, h] = tmp.prob[a, i, m]*V[a,h]
    @tullio tmp.gg_B2[i,l,m,h] = tmp.gg_BB[i, m, h]*delta_j[l, j, m]*tmp.KK[i, j, h]
    @tullio tmp.gg_B[l,m,h] = tmp.gg_B2[i,l,m,h]
    @tullio tmp.grad_v1[l, m, h] = tmp.gg_A[l, m, h] + tmp.gg_B[l, m, h]

    @tullio tmp.gg_C[i,l,m,h] = tmp.KK[i, j, h]*(V[Z[i, m], h]*delta_j[l, j, m]+V[Z[j, m], h]*delta_j[l, i, m])
    @tullio tmp.grad_v2[l, m, h] = tmp.gg_C[i,l,m,h]    
    @tullio tmp.grad_V[l, h] = _w[m]*(tmp.grad_v1[l, m, h] - tmp.grad_v2[l, m, h])
    
    @tullio tmp.reg_v01[i,j,l,h] = tmp.J[i,j,a,l]*V[a,h]
    @tullio tmp.reg_v02[i,j,l,h] = tmp.J[i,j,l,b]*V[b,h]
    @tullio tmp.reg_v1[i,l,h] = K[i, j, h]*tmp.reg_v01[i,j,l,h]
    @tullio tmp.reg_v2[i,l,h] = K[i, j, h]*tmp.reg_v02[i,j,l,h]
    @tullio tmp.reg_v[l, h] = tmp.reg_v1[i, l, h] + tmp.reg_v2[i, l, h]
    
    @tullio tmp.tot_grad_V[l, h] = tmp.grad_V[l, h] + 2*lambda*tmp.reg_v[l,h]

    return  (K=tmp.tot_grad_K, V=tmp.tot_grad_V)
end




function trainer_J(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)

    T = eltype(plmvar.K)

    m = if init_m !== Nothing
        init_m
    else
        (K = T.(init_fun(N, N, H)), V = T.(init_fun(q,H)))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
        
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->get_loss_J(x.K, x.V, z, _w; lambda = λ), m)[1]
            update!(t,m,g)
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i], lr[i] = get_loss_J_parts(m.K, m.V, plmvar.Z, _w; lambda = λ)
        p[i] = round((PPV[N]),digits=3)
        p2[i] = round((PPV[2*N]),digits=3)
        
        pi = p[i]
        p2i = p2[i]
        
        ltoti = round(l[i] + lr[i], digits=3)
        
        println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
    end
    
    
    if fig_save
        println("plotting figure")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    
        # Plot on the first subplot
        ax1.loglog(l .+ lr, label="tot loss")
        ax1.loglog(l, label="loss like")
        ax1.loglog(lr, label="loss reg")
        ax1.legend()
    
        # Plot on the second subplot
        id = ones(n_epochs)
        ax2.loglog(id, label="Ideal model")
        ax2.loglog(p, label="PPV@L")
        ax2.loglog(p2, label="PPV@2L")
        ax2.legend()

        savefig("log_J/H$(H)η$(η)λ$(λ)T$(n_epochs).png")    
    end
    n = (ltot = l.+lr, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return m, n
end



function trainer(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)

    MM = size(plmvar.Z,2) % batch_size

    delta_j_small = zeros(q,N,MM)
    delta_j = zeros(q,N,batch_size)
    println("updated")
    T = eltype(plmvar.K)

    tmp = Stg(plmvar; m = batch_size)
    tmp_tot = Stg(plmvar)
    tmp1 = Stg(plmvar; m = MM)
   
    m = if init_m !== Nothing
        init_m
    else
        (K = T.(init_fun(N, N, H)), V = T.(init_fun(q,H)))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            
            _w = w/sum(w)
            if length(_w) != batch_size
                @tullio delta_j_small[a, j, m] = a == z[j, m] (a in 1:q)
                g = get_anal_grad(m.K, m.V, z, _w, tmp1, delta_j_small; lambda = λ)
                update!(t,m,g)
            else
                @tullio delta_j[a, j, m] = a == z[j, m] (a in 1:q)
                g = get_anal_grad(m.K, m.V, z, _w, tmp, delta_j; lambda = λ)
                update!(t,m,g)
            end
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i], lr[i] = get_loss_parts(m.K, m.V, plmvar.Z, _w, tmp_tot; lambda = λ)
        #l[i], lr[i] = get_loss_J_parts(m.K, m.V, plmvar.Z, _w; lambda = λ)
        p[i] = round((PPV[N]),digits=3)
        p2[i] = round((PPV[2*N]),digits=3)
               
        ltoti = round(l[i] + lr[i], digits=3)
        
        println("Epoch $i ltot = $ltoti \t PPV@L = $(p[i]) \t PPV@2L = $(p2[i]) \t 1st Err = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $(p[i]) \t PPV@2L = $(p2[i]) \t 1st Err = $(findfirst(x->x!=1, PPV))")
    end
    
    if fig_save
        println("plotting figure")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    
        # Plot on the first subplot
        ax1.loglog(l .+ lr, label="tot loss")
        ax1.loglog(l, label="loss like")
        ax1.loglog(lr, label="loss reg")
        ax1.legend()
    
        # Plot on the second subplot
        id = ones(n_epochs)
        ax2.loglog(id, label="Ideal model")
        ax2.loglog(p, label="PPV@L")
        ax2.loglog(p2, label="PPV@2L")
        ax2.legend()

        savefig("log/H$(H)η$(η)λ$(λ)T$(n_epochs).png")    
    end
    
    n = (ltot = l.+lr, p = p, p2 = p2)
    
    savefile !== nothing && close(file)
    return m, n
end