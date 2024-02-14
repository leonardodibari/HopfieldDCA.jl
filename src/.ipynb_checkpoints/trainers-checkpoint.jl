function trainer_J(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.5,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    orthog=false,
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder = "../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false)
    
    if orthog == false
        foo = get_loss_J
        foos = get_loss_J_parts
    else
        println("orthogonalizaton applied on V")
        foo = get_loss_J_orthog
        foos = get_loss_J_orthog_parts
    end
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    savefile !== nothing && (savef = joinpath(savefile, "ort$(orthog)logH$(H)η$(η)λ$(λ)T$(n_epochs).txt"))
    
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
    savefile !== nothing && (println(savef))
        
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->foo(x.K, x.V, z, _w; lambda = T(λ)), m)[1]
            update!(t,m,g)
        end
        if i % 10 == 0
            _w = plmvar.W/sum(plmvar.W)
            s = score(m.K,m.V)
            PPV = compute_PPV(s,structfile)
        
            l[i], lr[i] = foos(m.K, m.V, plmvar.Z, _w; lambda = T(λ))
            p[i] = round((PPV[N]),digits=3)
            p2[i] = round((PPV[2*N]),digits=3)
        
            pi = p[i]
            p2i = p2[i]
        
            ltoti = round(l[i] + lr[i], digits=3)
        
            println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
            savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
        end
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
        println(joinpath(folder,"ort$(orthog)_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))
        savefig(joinpath(folder,"ort$(orthog)_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))  
    end
    
    if par_save
        s = HopfieldDCA.score(m.K, m.V); PPV_h = HopfieldDCA.compute_PPV(s, structfile);
        @load joinpath(folder, "PPV_s.jld2") ss
        @load joinpath(folder, "PPV_plm.jld2") splm
        
        @save joinpath(folder, "ort$(orthog)_parsH$(H)η$(η)λ$(λ)T$(n_epochs).jld2") m PPV_h
                
        fig,ax1 = plt.subplots()         
        ax1.plot(ss, label="Stucture")
        ax1.plot(splm, label="PLM")
        ax1.plot(PPV_h, label="Hop")
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of Predictions")
        ax1.set_ylabel("PPV")
        ax1.legend()
        savefig(joinpath(folder,"ort$(orthog)_ppv_H$(H)η$(η)λ$(λ)T$(n_epochs).png")) 
    end
    n = (ltot = l.+lr, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return m, n
end


function trainer_J_fixedV(plmvar, n_epochs, V; 
    batch_size = 1000,
    η = 0.5,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand,
    orthog = false,
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder = "../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    if orthog == false
        foo = get_loss_J
        foos = get_loss_J_parts
    else
        foo = get_loss_J_orthog
        foos = get_loss_J_orthog_parts
    end
    
    savefile !== nothing && (savef = joinpath(savefile, "fixVort$(orthog)_logH$(H)η$(η)λ$(λ)T$(n_epochs).txt"))
    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)

    T = eltype(plmvar.K)

    K = T.(init_fun(N, N, H))
    
    t = setup(Adam(η), K)

    savefile !== nothing && (file = open(savef,"a"))
    savefile !== nothing && (println(savef))
        
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(p1->foo(p1, V, z, _w; lambda = T(λ)), K)[1]
            update!(t,K,g)
        end
        if i % 10 == 0
            _w = plmvar.W/sum(plmvar.W)
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
        
            l[i], lr[i] = foos(K, V, plmvar.Z, _w; lambda = T(λ))
            p[i] = round((PPV[N]),digits=3)
            p2[i] = round((PPV[2*N]),digits=3)
        
            pi = p[i]
            p2i = p2[i]
        
            ltoti = round(l[i] + lr[i], digits=3)
        
            println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
            savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
        end
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
        println(joinpath(folder,"fixVort$(orthog)_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))
        savefig(joinpath(folder,"fixVort$(orthog)_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))  
    end
    
    if par_save
        s = HopfieldDCA.score(K, V); PPV_h = HopfieldDCA.compute_PPV(s, structfile);
        @load joinpath(folder, "PPV_s.jld2") ss
        @load joinpath(folder, "PPV_plm.jld2") splm
        
        @save joinpath(folder, "fixVort$(orthog)_parsH$(H)η$(η)λ$(λ)T$(n_epochs).jld2") K PPV_h
                
        fig,ax1 = plt.subplots()         
        ax1.plot(ss, label="Stucture")
        ax1.plot(splm, label="PLM")
        ax1.plot(PPV_h, label="Hop")
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of Predictions")
        ax1.set_ylabel("PPV")
        ax1.legend()
        savefig(joinpath(folder,"fixVort$(orthog)_ppv_H$(H)η$(η)λ$(λ)T$(n_epochs).png")) 
    end
    n = (ltot = l.+lr, p = p, p2 = p2)
    savefile !== nothing && close(file)
    return K, n
end


function trainer_fullJ(plmvar, n_epochs; 
    batch_size = 1000,
    η = 0.5,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder ="../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false)
    
    D = (plmvar.Z, plmvar.W)
    H = plmvar.H
    N = plmvar.N
    q = plmvar.q

    savefile !== nothing && (savef = joinpath(savefile, "logfull_H$(H)η$(η)λ$(λ)T$(n_epochs).txt"))
    
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

    savefile !== nothing && (file = open(savef,"a"))
    savefile !== nothing && (println(savef))
    
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->get_loss_fullJ(x.K, x.V, z, _w; lambda = T(λ)), m)[1]
            update!(t,m,g)
        end
        
        if i % 10 == 0
            _w = plmvar.W/sum(plmvar.W)
            s = score_full(m.K,m.V)
            PPV = compute_PPV(s,structfile)
        
            l[i], lr[i] = get_loss_fullJ_parts(m.K, m.V, plmvar.Z, _w; lambda = T(λ))
            p[i] = round((PPV[N]),digits=3)
            p2[i] = round((PPV[2*N]),digits=3)
        
            pi = p[i]
            p2i = p2[i]
        
            ltoti = round(l[i] + lr[i], digits=3)
        
            println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
            savefile !== nothing && println(file, "Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
        end
    end
    
    
    if fig_save
        println("plotting figure")
        fig, (ax1, ax2) = plt.subplots()
    
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
        println(joinpath(folder,"full_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))
        savefig(joinpath(folder,"full_H$(H)η$(η)λ$(λ)T$(n_epochs).png"))  
    end
    
    
    if par_save
        s = HopfieldDCA.score(m.K, m.V); PPV_h = HopfieldDCA.compute_PPV(s, structfile);
        @load joinpath(folder,"PPV_s.jld2") ss
        @load joinpath(folder,"PPV_plm.jld2") splm
        @save joinpath(folder,"parsfull_H$(H)η$(η)λ$(λ)T$(n_epochs).jld2") m, PPV_h
        
        fig,ax1 = plt.subplots()
        ax1.plot(ss, label="Stucture")
        ax1.plot(splm, label="PLM")
        ax1.plot(PPV_h, label="fullHop")
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of Predictions")
        ax1.set_ylabel("PPV")
        ax1.legend()
        savefig(joinpath(folder,"ppvfull_H$(H)η$(η)λ$(λ)T$(n_epochs).png")) 
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