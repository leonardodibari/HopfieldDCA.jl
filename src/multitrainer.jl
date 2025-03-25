function multi_loss(Ks, V, hs, Zs, Ws; lambda = lambda, orthog = false, ort = 50)
    
    if orthog == true
        foo = get_loss_J_orthog
    else
        foo = get_loss_J
    end
    
    
    Nf = length(Ks)
    T = eltype(lambda)
    tot_loss = T(0.0)
    for i in 1:Nf
        tot_loss = tot_loss + foo(Ks[i], V, hs[i], Zs[i], Ws[i], lambda=lambda[i], ort = ort)
    end

    return tot_loss
end



function multitrainer(fs::Vector{Int}, plmvars, n_epochs::Union{Int,Vector{Int}};
    init_m = Nothing,
    η = 0.5, 
    n_batches = 50, 
    ort = 50,
    init = rand,
    orthog = false,
    each_step = 10,
    lambda::Union{T,Vector{T}}=fill(0.001, length(plmvars)), 
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false, 
    plm_ppv = true) where {T}
    
    if orthog == false
        foo = get_loss_J
    else
        println("orthogonalizaton applied on V, using multiloss2 corrected")
        foo = get_loss_J_orthog
    end
    
    NF = length(plmvars)
    TT = eltype(plmvars[1].K)
    
    D = [(plmvars[f].Z, plmvars[f].W) for f in 1:NF]
    
    lambda = TT.(lambda)
    if typeof(lambda) == TT
        lambda = fill(lambda,NF)
    end

    if typeof(n_epochs) == Int
        n_epochs = fill(n_epochs, NF)
    end

    
    #creazione arvar per ogni famiglia
    q = maximum(D[1][1])
    H = plmvars[1].H
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
    end
    
    println(Ns)
    m = if init_m !== Nothing
        init_m
    else
        (Ks = init.(TT, Ns, Ns, H), hs = init.(TT, q, Ns), V = init(TT, q, H))
    end
    
    println(size(m.hs[1]))

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    loaders = Vector{Any}(undef, NF)
    for i in 1:maximum(n_epochs)
        flags = i .<= n_epochs
        for n in 1:NF
            loaders[n] = DataLoader(D[n], batchsize = batch_sizes[n], shuffle = true)
        end
        loader = zip(loaders...)
        for pf in loader
            ws = [pf[m][2]/sum(pf[m][2]) for m in 1:NF]
            Zs = [pf[m][1] for m in 1:NF] 
            g = gradient(x->multi_loss(x.Ks[flags], x.V, x.hs[flags], Zs[flags], ws[flags], lambda = lambda[flags], 
                    orthog = orthog, ort = ort),m)[1]
            update!(t,m,g)
        end

        if i % each_step == 0
            losses = [round(foo(m.Ks[n], m.V, m.hs[n], D[n][1], D[n][2], 
                        lambda=lambda[n],  ort = ort),digits=2) for n in 1:NF]
            print("Epoch $i ") 
            [print("PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            println("-> Total loss = $(round(sum(losses),digits=3))")
            savefile !== nothing && print(file, "Epoch $i ") 
            savefile !== nothing && [print(file, "PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            savefile !== nothing && println(file, "-> Total loss = $(round(sum(losses),digits=3))")
            for i in 1:length(fs) 
                s = HopfieldDCA.score(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);
                print("PF$(fs[i]) = $(round(PPV_h[Ns[i]], digits = 3)), ")
                savefile !== nothing && print(file, "PF$(fs[i]) = $(round(PPV_h[Ns[i]], digits = 3)), ")
            end
            println("-> PPV @L")
            savefile !== nothing && println(file, "-> PPV @L")
            end
    end
    savefile !== nothing && close(file)
        
    if fig_save == true
        out_path ="../multi_fam/ort$(orthog)_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);
            
            fig,ax1 = plt.subplots()  
            if plm_ppv == true
                @load joinpath(folders_dict[fs[i]], "PPV_s.jld2") ss
                @load joinpath(folders_dict[fs[i]], "PPV_plm.jld2") splm
                ax1.plot(ss, label="Stucture")
                ax1.plot(splm, label="PLM")
            end
            ax1.plot(PPV_h, label="Hop")
            ax1.set_xscale("log")
            ax1.set_xlabel("Number of Predictions")
            ax1.set_ylabel("PPV")
            ax1.legend()
            savefig(joinpath(out_path,"mf_ppv$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).png")) 
        end        
    end
    
    if par_save == true 
        out_path ="../multi_fam/ort$(orthog)_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);   
            @save joinpath(out_path, "mf_pars$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).jld2") m PPV_h
        end
    end
        
    return m
end

function multi_loss_full(Ks, V, hs, Zs, Ws; lambda = lambda, orthog = false, ort = 50)
    
    if orthog == true
        foo = get_loss_fullJ_orthog
    else
        foo = get_loss_fullJ
    end
    
    
    Nf = length(Ks)
    T = eltype(lambda)
    tot_loss = T(0.0)
    for i in 1:Nf
        tot_loss = tot_loss + foo(Ks[i], V, hs[i], Zs[i], Ws[i], lambda=lambda[i], ort = ort)
    end

    return tot_loss
end


function multitrainer_full(fs::Vector{Int}, plmvars, n_epochs::Union{Int,Vector{Int}};
    init_m = Nothing,
    η = 0.5, 
    n_batches = 50, 
    ort = 50,
    init = rand,
    orthog = false,
    each_step = 10,
    lambda::Union{T,Vector{T}}=fill(0.001, length(plmvars)), 
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false, 
    plm_ppv = true) where {T}
    
    if orthog == false
        foo = get_loss_fullJ
    else
        println("orthogonalizaton applied on V, using multiloss2 corrected")
        foo = get_loss_fullJ_orthog
    end
    
    NF = length(plmvars)
    TT = eltype(plmvars[1].K)
    
    D = [(plmvars[f].Z, plmvars[f].W) for f in 1:NF]
    
    lambda = TT.(lambda)
    if typeof(lambda) == TT
        lambda = fill(lambda,NF)
    end

    if typeof(n_epochs) == Int
        n_epochs = fill(n_epochs, NF)
    end

    
    q = maximum(D[1][1])
    H = plmvars[1].H
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
    end
    
    println(Ns)
    m = if init_m !== Nothing
        init_m
    else
        (Ks = init.(TT, Ns, H), V = init(TT, q, H), hs = init.(TT, q, Ns))
    end

    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savefile,"a"))
    
    loaders = Vector{Any}(undef, NF)
    for i in 1:maximum(n_epochs)
        flags = i .<= n_epochs
        for n in 1:NF
            loaders[n] = DataLoader(D[n], batchsize = batch_sizes[n], shuffle = true)
        end
        loader = zip(loaders...)
        for pf in loader
            ws = [pf[m][2]/sum(pf[m][2]) for m in 1:NF]
            Zs = [pf[m][1] for m in 1:NF] 
            g = gradient(x->multi_loss_full(x.Ks[flags], x.V, x.hs[flags], Zs[flags], ws[flags], 
                    lambda =lambda[flags], orthog = orthog, ort = ort),m)[1]
            update!(t,m,g)
        end

        if i % each_step == 0
            losses = [round(foo(m.Ks[n], m.V, m.hs[n], D[n][1], D[n][2], 
                        lambda=lambda[n],  ort = ort),digits=2) for n in 1:NF]
            print("Epoch $i ") 
            [print("PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            println("-> Total loss = $(round(sum(losses),digits=3))")
            savefile !== nothing && print(file, "Epoch $i ") 
            savefile !== nothing && [print(file, "PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            savefile !== nothing && println(file, "-> Total loss = $(round(sum(losses),digits=3))")
            for i in 1:length(fs) 
                s = HopfieldDCA.score_full(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);
                print("PF$(fs[i]) = $(round(PPV_h[Ns[i]], digits = 3)), ")
                savefile !== nothing && print(file, "PF$(fs[i]) = $(round(PPV_h[Ns[i]], digits = 3)), ")
            end
            println("-> PPV @L")
            savefile !== nothing && println(file, "-> PPV @L")
            end
    end
    savefile !== nothing && close(file)
        
    if fig_save == true
        out_path ="../multi_fam/full_ort$(orthog)_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score_full(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);
            
            fig,ax1 = plt.subplots()  
            if plm_ppv == true
                @load joinpath(folders_dict[fs[i]], "PPV_s.jld2") ss
                @load joinpath(folders_dict[fs[i]], "PPV_plm.jld2") splm
                ax1.plot(ss, label="Stucture")
                ax1.plot(splm, label="PLM")
            end
            ax1.plot(PPV_h, label="Hop")
            ax1.set_xscale("log")
            ax1.set_xlabel("Number of Predictions")
            ax1.set_ylabel("PPV")
            ax1.legend()
            savefig(joinpath(out_path,"full_mf_ppv$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).png")) 
        end        
    end
    
    if par_save == true 
        out_path ="../multi_fam/full_ort$(orthog)_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score_full(m.Ks[i], m.V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);   
            @save joinpath(out_path, "full_mf_pars$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).jld2") m PPV_h
        end
    end
        
    return m
end


function multitrainer_fixedV(fs::Vector{Int}, plmvars, n_epochs::Union{Int,Vector{Int}}, V::Matrix{T};
    init_m = Nothing,
    η = 0.5, 
    n_batches = 50, 
    init = rand,
    lambda::Union{T,Vector{T}}=fill(0.001, length(plmvars)), 
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false) where {T}
    
    NF = length(plmvars)
    TT = eltype(plmvars[1].K)
    
    D = [(plmvars[f].Z, plmvars[f].W) for f in 1:NF]
    
    lambda = TT.(lambda)
    if typeof(lambda) == TT
        lambda = fill(lambda,NF)
    end

    if typeof(n_epochs) == Int
        n_epochs = fill(n_epochs, NF)
    end
    
    println("fixed V")
    
    #creazione arvar per ogni famiglia
    q = maximum(D[1][1])
    H = plmvars[1].H
    Ns = zeros(Int, NF)
    Ms = zeros(Int, NF)
    batch_sizes = zeros(Int, NF)
    for i in 1:NF
        Ns[i], Ms[i] = size(D[i][1])
        batch_sizes[i] = Int(round(Ms[i]/n_batches))
    end

    Ks = init.(TT, Ns, Ns, H)
    
    t = setup(Adam(η), Ks)

    savefile !== nothing && (file = open(savefile,"a"))
    
    loaders = Vector{Any}(undef, NF)
    for i in 1:maximum(n_epochs)
        flags = i .<= n_epochs
        for n in 1:NF
            loaders[n] = DataLoader(D[n], batchsize = batch_sizes[n], shuffle = true)
        end
        loader = zip(loaders...)
        for pf in loader
            ws = [pf[m][2]/sum(pf[m][2]) for m in 1:NF]
            Zs = [pf[m][1] for m in 1:NF] 
            g = gradient(p1->multi_loss(p1[flags], V, Zs[flags], ws[flags], lambda = lambda[flags]),Ks)[1]
            update!(t,Ks,g)
        end

        if i % 10 == 0
            print("Epoch $i ") 
            losses = [round(get_loss_J(Ks[i], V, D[i][1], D[i][2],  lambda=lambda[i]),digits=2) for i in 1:NF]
            [print("PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            println("-> Total loss = $(round(sum(losses),digits=3))")
            savefile !== nothing && print(file, "Epoch $i ") 
            savefile !== nothing && [print(file, "PF$(fs[n]) = $(losses[n]), ") for n in 1:NF]
            savefile !== nothing && println(file, "-> Total loss = $(round(sum(losses),digits=3))")
            savefile !== nothing && println(file, "total loss = $(round(sum(losses),digits=3)))") 
        end
    end
    savefile !== nothing && close(file)
        
    if fig_save == true
        out_path ="../multi_fam/fixV_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score(Ks[i], V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);
            @load joinpath(folders_dict[fs[i]], "PPV_s.jld2") ss
            @load joinpath(folders_dict[fs[i]], "PPV_plm.jld2") splm
    
            fig,ax1 = plt.subplots()         
            ax1.plot(ss, label="Stucture")
            ax1.plot(splm, label="PLM")
            ax1.plot(PPV_h, label="Hop")
            ax1.set_xscale("log")
            ax1.set_xlabel("Number of Predictions")
            ax1.set_ylabel("PPV")
            ax1.legend()
            savefig(joinpath(out_path,"mf_fixV_ppv$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).png")) 
        end        
    end
    
    if par_save == true 
        out_path ="../multi_fam/fixV_H$(H)η$(η)λ$(lambda[1])/" 
        
        if !isdir(out_path)
            println("creating directory")
            mkdir(out_path)
        end

        for i in 1:length(fs) 
            s = HopfieldDCA.score(Ks[i], V); PPV_h = HopfieldDCA.compute_PPV(s, structs_dict[fs[i]]);   
            @save joinpath(out_path, "mf_fixVpars$(fs[i])_H$(H)η$(η)λ$(lambda[1])T$(n_epochs[i]).jld2") Ks PPV_h
        end
    end
        
    return Ks
end
