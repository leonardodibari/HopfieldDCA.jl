function get_loss_hop(xi::Array{T,3}, 
    K::Array{T,2},
    V::Array{T,2}; 
    lambda::T = T(0.001)) where {T}
    
    @tullio csi[h, a, i] := K[i, h] * V[a, h]
    @tullio loss[h, i, a] := xi[h, a, i] - csi[h, a, i]
    
    return sum(abs2, loss) + lambda * (sum(abs2, K) + sum(abs2, V))
    
end


function get_loss_hop_parts(xi::Array{T,3}, 
    K::Array{T,2},
    V::Array{T,2}; 
    lambda::T = T(0.001)) where {T}
    
    @tullio csi[h, a, i] := K[i, h] * V[a, h]
    @tullio loss[h, i, a] := xi[h, a, i] - csi[h, a, i]
    
    return round(sum(abs2, loss), digits = 3), round(lambda * (sum(abs2, K) + sum(abs2, V)), digits = 3)
    
end

function score_hopf(xi; min_separation::Int=6)

    H = size(xi,1)
    L = size(xi, 3)
    q = 20 
    
    @tullio Jtens[a, b, i, j] := xi[h, a, i] * xi[h, b, j] *(j != i) 

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end

function trainer_hop(n_epochs, xi; 
    η = 0.5,
    λ = 0.001,
    structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    init_m = Nothing, 
    init_fun = rand, 
    par_save = false)
    
    T = eltype(xi)
    
    l = zeros(n_epochs)
    lr = zeros(n_epochs)
    p = zeros(n_epochs)
    p2 = zeros(n_epochs)
    
    N = size(xi,3)
    H = size(xi, 1)
    q = 20


    m = if init_m !== Nothing
        init_m
    else
        (K = T.(init_fun(N, H)), V = T.(init_fun(q,H)))
    end
    
    
    for i in 1:n_epochs
        g = gradient(x->get_loss_hop(xi, x.K, x.V; lambda = T(λ)), m)[1]
        m.K .-= η * g.K
        m.V .-= η * g.V
        if i % 50 == 0
            s = score_full(m.K,m.V)
            PPV = compute_PPV(s,structfile)
            l[i], lr[i] = get_loss_hop_parts(xi, m.K, m.V; lambda = T(λ))
            p[i] = round((PPV[N]),digits=3)
            p2[i] = round((PPV[2*N]),digits=3)
        
            pi = p[i]
            p2i = p2[i]
        
            ltoti = round(l[i] + lr[i], digits=3)
            
            #println("Epoch $i ltot = $ltoti lreg = $(lr[i])")
            println("Epoch $i ltot = $ltoti \t PPV@L = $pi \t PPV@2L = $p2i \t 1st Err = $(findfirst(x->x!=1, PPV))")
            
        end
    end
    
    s = score_full(m.K,m.V)
    return m, s
end
