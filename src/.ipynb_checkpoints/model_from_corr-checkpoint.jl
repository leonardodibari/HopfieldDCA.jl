
function opt_loss_ij(c_inv_ij::Array{T,2}, K_ij::Vector{T}, V::Array{T,2}) where {T}
    
    # Compute J using Tullio
    @tullio J[a, b] := K_ij[h] * V[a, h] * V[b, h]
    
    # Return the loss
    return sum(abs2, c_inv_ij .+ J)
end

function opt_grad_ij!(c_inv_ij::Array{T,2}, K_ij::Vector{T}, V::Array{T,2}) where {T}
    
    # Compute J using Tullio
    @tullio J[a, b] := K_ij[h] * V[a, h] * V[b, h]
    @tullio VV[a,b,h] := V[a, h] * V[b, h]
    
    # Return the loss
    return T.([sum(2 .*(c_inv_ij .+ J).*VV[:,:,head]) for head in 1:20])
end

function fg!(F, G, K_ij::Vector{T}, c_inv_ij::Array{T,2}, V::Array{T,2}) where {T}
    @tullio J[a, b] := K_ij[h] * V[a, h] * V[b, h]
    @tullio VV[a,b,h] := V[a, h] * V[b, h]
    
    if G !== nothing
        G .= [sum(2 .*(c_inv_ij .+ J).*VV[:,:,head]) for head in 1:20]
    end
    
    if F !== nothing
        return sum(abs2, c_inv_ij .+ J)
    end
end
            
        
        


function minimize_opt_loss_ij(Z::Array{Int,2}, 
    V::Array{T,2}, 
    N::Int,
    H::Int;
    q::Int = 21,
    iterations::Int=5, 
    alpha = 0.5,
    tol=1e-10,
    optimizer=LBFGS())  where {T}
    
    V_copy = copy(V[1:q-1, 1:H-1]); 
    #V_copy = copy(V);
    #V_copy[:,21] .-= 1.
    K0 = T.(rand(N, N, H-1));
    for i in 1:N
        K0[i,i,:] .= T(0);
    end
    K_opt = deepcopy(K0);
    # Get dimensions of K
    dims = size(K0)
    
    ff1,ff2 = compute_weighted_frequencies(Int8.(Z),q, 0.2);
    f1, f2 = add_pseudocount(ff1,ff2,alpha, q);
    c_inv = T.(reshape(inv(f2 - f1 * f1'), q-1, N, q-1, N));
    
    ff1,ff2 = compute_weighted_frequencies(Int8.(Z),q+1, 0.2);
    f1_tot, f2_tot = add_pseudocount(ff1,ff2,alpha, q+1);
    f1_tot_rs = reshape(f1_tot, q, N);
    
    
    @time for i in 1:N
        for j in 1:N
            c_inv_ij = c_inv[:,i,:,j] 
            if i == j
                #c_inv_ij .-= (1/f1_tot_rs[q])
               # for a in 1:q-1
                #    c_inv_ij[a,a] -= 1/f1_tot_rs[a]
                #end
            else
            
                K_vec_init = K0[i,j,:];
                #objective(K_vec) = opt_loss_ij(c_inv_ij, K_vec, V_copy)
                #result = optimize(objective, K_vec_init, optimizer, Optim.Options(f_tol = tol));
                
                wrapfg! = (F, G, K_vec) -> fg!(F , G, K_vec, c_inv_ij, V_copy)
                result = optimize(Optim.only_fg!(wrapfg!), K_vec_init, optimizer, Optim.Options(f_tol = tol));
                
                K_opt[i,j,:] .= result.minimizer
            end
    
        end
    end
   
    return K_opt, V_copy
    
end




function get_corr_mf(Z::Array{Int,2},
    K::Array{T,3},
    V::Array{T,2}, 
    N::Int,
    H::Int;
    alpha = 0.5,
    q::Int = 21)  where {T}
    
     
    
    ff1,ff2 = compute_weighted_frequencies(Int8.(Z),q, 0.2);
    f1, f2 = add_pseudocount(ff1,ff2,alpha, q);
    f1_rs = reshape(f1, q-1, N);
    
    c_inv = T.(reshape(inv(f2 - f1 * f1'), q-1, N, q-1, N));
    
    #=ff1,ff2 = compute_weighted_frequencies(Int8.(Z),q+1, 0.2);
    f1_tot, f2_tot = add_pseudocount(ff1,ff2,alpha, q+1);
    f1_tot_rs = reshape(f1_tot, q, N);=#
    
    @tullio J[a,i,b,j] := K[i,j,h]*V[a,h]*V[b,h] * (i != j)
    
    c_mf_inv = zeros(q-1, N, q-1,N);
    
    @time for i in 1:N
        for j in 1:N 
            if i == j
                c_mf_inv[:,i,:,i] .+= c_inv[:,i,:,i]   #(1/(1-sum(f1_rs[:,i])))
                #=for a in 1:q-1
                    c_mf_inv[a,i,a,i] += (1/f1_rs[a,i])
            end=#
                
            else 
                c_mf_inv[:,i,:,j] .-= J[:,i,:,j]
                println(i," ", j, " ", sum(abs2, c_mf_inv[:,i,:,j] .- c_inv[:,i,:,j]), " ", 
                    cor(c_mf_inv[:,i,:,j][:], c_inv[:,i,:,j][:]))
            end
            #println(i," ", j, " ", sum(abs2, c_mf_inv[:,i,:,j] .- c_inv[:,i,:,j]))
        end
    end
    
    println(cor(c_mf_inv[:], c_inv[:]))
    c_emp = f2 - f1 * f1'
    c_opt_inv = reshape(c_mf_inv, (q-1)*N, (q-1)*N)
        
    return c_emp, inv(c_emp), inv(c_opt_inv), c_opt_inv
end


function small_trial(Z::Array{Int,2},
    N::Int;
    alpha = 0.5,
    q::Int = 21)  
     
    
    ff1,ff2 = compute_weighted_frequencies(Int8.(Z),q, 0.2);
    f1, f2 = add_pseudocount(ff1,ff2,alpha, q);
    c_emp = Float64.(reshape(f2 - f1 * f1', q-1, N, q-1, N));
    c_inv = Float64.(reshape(inv(f2 - f1 * f1'), q-1, N, q-1, N));
    

    c_inv2 = zeros(q-1,N,q-1,N)
    @time for i in 1:N
        for j in 1:N 
            println((i, j))
            c_inv2[:,i,:,j] .= inv(c_emp[:,i,:,j])
        end 
    end
    
    cor(c_inv2[:], c_inv[:])
    
    return c_inv, c_inv2
end

#=
function trainer_J_givenV(plmvar, n_epochs, V; 
    batch_size = 1000,
    η = 0.5,
    λ = 0.001,
    init_m = Nothing, 
    init_fun = rand, 
    orthog=false,
    ar = false,
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    folder = "../DataAttentionDCA/data/PF00014/",
    savefile::Union{String, Nothing} = nothing,
    fig_save = false,
    par_save = false)
    
  
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
        (K = T.(init_fun(N, N, H)), V = T.(V), h = T.(init_fun(q,N)))
    end
    
    mj = V[1:20, 1:20]
    
    if ar == true
        @tullio m.K[i,j,h] = m.K[i,j,h] * (j<i)
    end
    
    t = setup(Adam(η), m)

    savefile !== nothing && (file = open(savef,"a"))
    savefile !== nothing && (println(savef))
        
    for i in 1:n_epochs
        loader = DataLoader(D, batchsize = batch_size, shuffle = true)
        for (z,w) in loader
            _w = w/sum(w)
            g = gradient(x->foo(x.K, x.V, x.h, z, _w; lambda = T(λ)), m)[1]
            update!(t,m,g)
            if ar == true
                @tullio m.K[i,j,h] = m.K[i,j,h] * (j<i)
            end
            m.V[1:20, 1:20] .= mj
        end
        if i % 10 == 0
            _w = plmvar.W/sum(plmvar.W)
            s = score(m.K,m.V)
            PPV = compute_PPV(s,structfile)
        
            l[i], lr[i] = foos(m.K, m.V, m.h, plmvar.Z, _w; lambda = T(λ))
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

=#