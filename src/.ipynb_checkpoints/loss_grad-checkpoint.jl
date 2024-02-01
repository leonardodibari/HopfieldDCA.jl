function get_loss_J(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    Z::Array{Int,2}, 
    _w::Array{Float64, 1}; lambda = 0.001)
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    return sum(loss) + lambda * sum(abs2, J)
end

function get_loss_J_parts(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    Z::Array{Int,2}, 
    _w::Array{Float64, 1}; lambda = 0.001)
    
    @tullio KK[i,j,h] := K[i,j,h]*(j!=i)
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    return round(sum(loss), digits = 3), round(lambda * sum(abs2, J), digits = 3) 
end

function get_loss_J_tmp(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    Z::Array{Int,2}, 
    _w::Array{Float64, 1}, tmp; lambda = 0.001)
    
    @tullio tmp.KK[i,j,h] = K[i,j,h]*(j!=i)
    @tullio tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    @tullio tmp.en[a, i, m] = tmp.J[i, j, a, Z[j, m]]
    @tullio tmp.data_en[i, m] = tmp.en[Z[i, m], i, m]
    tmp.log_z = logsumexp(tmp.en)[1,:,:]
    @tullio tmp.loss[i] = _w[m]*(tmp.log_z[i, m] - tmp.data_en[i,m])
    
    return sum(tmp.loss) + lambda * sum(abs2, tmp.J)
end

function get_loss_J_tmp_parts(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    Z::Array{Int,2}, 
    _w::Array{Float64, 1}, tmp; lambda = 0.001)
    
    @tullio tmp.KK[i,j,h] = K[i,j,h]*(j!=i)
    @tullio tmp.J[i,j,a,b] = tmp.KK[i,j,h]*V[a,h]*V[b,h]
    @tullio tmp.en[a, i, m] = tmp.J[i, j, a, Z[j, m]]
    @tullio tmp.data_en[i, m] = tmp.en[Z[i, m], i, m]
    tmp.log_z = logsumexp(tmp.en)[1,:,:]
    @tullio tmp.loss[i] = _w[m]*(tmp.log_z[i, m] - tmp.data_en[i,m])
    
    return round(sum(tmp.loss), digits=3), round(lambda * sum(abs2, tmp.J), digits = 3)
end

function get_loss(K::Array{Float64,3}, 
    V::Array{Float64,2}, 
    Z::Array{Int,2}, 
    _w::Array{Float64, 1}; lambdaK = 0.005, lambdaV = 0.005)
    
    
    @tullio J[i,j,a,b] := K[i,j,h]*V[a,h]*V[b,h]*(j!=i)
    @tullio en[a, i, m] := J[i, j, a, Z[j, m]]
    @tullio data_en[i, m] := en[Z[i, m], i, m]
    log_z = logsumexp(en)[1,:,:]
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
    
    #regularization
    @tullio reg_v := lambdaV*V[a, h]*V[a, h]
    @tullio sreg_k := lambdaK * K[i,j,h] * K[i,j,h] * (j!=i)
    #println("loss is $(sum(loss)) + $(sreg_k) + $reg_v and sumn=$(sum(en)) and sumn=$(sum(data_en))")
    return sum(loss) + sreg_k + reg_v  
end


function trainer_J(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)


    m = if init_m !== Nothing
        init_m
    else
        (K = init_fun(N, N, H), V = init_fun(q,H))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    tmp = StgArr(plmvar)
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->get_loss_J_tmp(x.K, x.V, z, _w, tmp; lambda = λ), m)[1]
            update!(t,m,g)
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i], lr[i] = get_loss_J_parts_tmp(m.K, m.V, plmvar.Z, _w, tmp; lambda = λ)
        p[i] = round((PPV[N]),digits=3)
        p2[i] = round((PPV[2*N]),digits=3)
        
        ltoti = round(l[i] + lr[i], digits=3)
        
        println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
    end
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))

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
    
    n = (ltot = l.+lr, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return m, n
end



function trainer(plmvar::HopPlmVar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    savefile::Union{String, Nothing} = nothing)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    
    l = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)


    m = if init_m !== Nothing
        init_m
    else
        (K = init_fun(N, N, H), V = init_fun(q,H))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->get_loss(x.K, x.V, z, _w; lambdaK = λ, lambdaV = λ), m)[1]
            #println(typeof(g))
            #println(size(g))
            update!(t,m,g)
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i] = round(get_loss(m.K, m.V, plmvar.Z, _w; lambdaK = λ, lambdaV = λ), digits = 4)
        p[i] = round((PPV[N]),digits=3)
        p2[i] = round((PPV[2*N]),digits=3)
        
        li = l[i]
        pi = p[i]
        p2i = p2[i]
        
        println("Epoch $i loss = $li \t PPV@L = $pi \t PPV@2L = $p2i \t First Error = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i loss = $li \t PPV@L = $pi \t PPV@2L = $p2i \t First Error = $(findfirst(x->x!=1, PPV))")
    end
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))

    ax1.loglog(l, label="loss")
    ax1.legend()

    # Plot on the second subplot
    ax2.loglog(p, label="PPV@L")
    ax2.loglog(p2, label="PPV@2L")
    ax2.legend()
    
    savefig("log/H$(H)η$(η)λ$(λ)T$(n_epochs).png")
    
    n = (l = l, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return m, n
end