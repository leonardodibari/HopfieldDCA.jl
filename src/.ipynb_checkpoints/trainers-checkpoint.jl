function trainer_J(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder = "../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    log_save = false,
    par_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    savefile !== nothing && (savef = joinpath(savefile, "H$(H)η$(η)λ$(λ)T$(n_epochs).png"))
    
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

    savefile !== nothing && (file = open(savef,"a"))
    println(savef)
        
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
        
        savefig(joinpath(folder,"H$(H)η$(η)λ$(λ)T$(n_epochs).png"))  
    end
    
    if par_save
        @save joinpath(folder,"H$(H)η$(η)λ$(λ)T$(n_epochs).jld2") m 
    end
    n = (ltot = l.+lr, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return m, n
end


function trainer_fullJ(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.005,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder ="../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    log_save = false,
    par_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    savefile !== nothing && (savefile = joinpath(savefile, "full_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))
    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)

    T = eltype(plmvar.K)

    m = if init_m !== Nothing
        init_m
    else
        (K = T.(init_fun(N, H)), V = T.(init_fun(q,H)))
    end
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
        
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->get_loss_fullJ(x.K, x.V, z, _w; lambda = λ), m)[1]
            update!(t,m,g)
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i], lr[i] = get_loss_fullJ_parts(m.K, m.V, plmvar.Z, _w; lambda = λ)
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
        
        savefig(joinpath(folder,"full_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))  
    end
    
    if par_save
        @save joinpath(folder,"full_H$(H)η$(η)λ$(λ)T$(n_epochs).jld2") m 
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
                g = get_anal_grad(m.K, m.V, z, _w, tmp1; lambda = λ)
                update!(t,m,g)
            else
                g = get_anal_grad(m.K, m.V, z, _w, tmp; lambda = λ)
                update!(t,m,g)
            end
        end
        _w = plmvar.W/sum(plmvar.W)
        s = score(m.K,m.V)
        PPV = compute_PPV(s,structfile)
        
        l[i], lr[i] = get_loss_parts(m.K, m.V, plmvar.Z, _w, tmp_tot; lambda = λ)
       
        p[i] = round((PPV[N]),digits=3)
        p2[i] = round((PPV[2*N]),digits=3)
               
        ltoti = round(l[i] + lr[i], digits=3)
        
        println("Epoch $i ltot = $ltoti \t PPV@L = $(p[i]) \t PPV@2L = $(p2[i]) \t 1st Err = $(findfirst(x->x!=1, PPV))")
        savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $(p[i]) \t PPV@2L = $(p2[i]) \t 1st Err = $(findfirst(x->x!=1, PPV))")
    end
    
    if fig_save
        
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