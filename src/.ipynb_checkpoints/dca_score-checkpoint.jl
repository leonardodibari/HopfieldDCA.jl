function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    if size(d,2) == 4 
        return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1) if d[i,4] != 0)
    elseif size(d,2) == 3
        Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,3] for i in 1:size(d,1) if d[i,3] != 0)
    end

end


function sel_good_res(K, V, filestruct; mindist::Int=6, ppv_cutoff=0.8)
    s = score(K, V, min_separation = mindist)
    dist = compute_residue_pair_dist(filestruct)
    good_ref_score = filter(x->x[4]>ppv_cutoff, compute_referencescore(s, dist, mindist = mindist))
    only_contacts = filter(x->x[5] == 1, good_ref_score)
    return map(x->(x[1], x[2]), only_contacts)
end

function sel_good_res(score,filestruct; mindist::Int=6, ppv_cutoff=0.8)
    dist = compute_residue_pair_dist(filestruct)
    good_ref_score = filter(x->x[4]>ppv_cutoff, compute_referencescore(score, dist, mindist = mindist))
    only_contacts = filter(x->x[5] == 1, good_ref_score)
    return map(x->(x[1], x[2]), only_contacts)
end

function epis_score(K, V, Z; q = 21, min_separation::Int=6)
    seq = Z[:,1]
    N = size(Z,1)
    @tullio J0[a, i, b, j] := K[i,j,h] * V[a,h] * V[b,h]
    JJ = zeros(size(J0)) 
    for a in 1:q
        for b in 1:q
            for i in 1:N
                for j in 1:N
                    a_i = seq[i]
                    b_j = seq[j]
                    JJ[a, i ,b, j] = J0[a, i, b, j] - J0[a, i, b_j, j] - J0[a_i, i, b, j] + J0[a_i, i, b_j, j]
                end
            end
        end
    end
    Jzsg = zsg(JJ)
    FN = compute_fn(Jzsg, N, q)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end         
        
    
    



function get_filt_mat_sf(K,Vn, filestruct; mindist::Int=6, ppv_cutoff=0.8, order_Martin = [1, 2, 5, 8, 10, 11, 18, 19, 20, 13, 7, 9, 15, 3, 4, 12, 14, 16, 17, 6])
    V = Vn[order_Martin,:]
    gr = sel_good_res(K, V, filestruct, mindist = mindist, ppv_cutoff = ppv_cutoff)
    N = size(K, 1)
    H = size(K, 3)
    KK = zeros(N,N,H)
    for n in 1:size(gr,1)
        for h in 1:21
            KK[gr[n][1], gr[n][2], h] = K[gr[n][1], gr[n][2], h]
        end
    end
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    J0 = mean(mean(J, dims = 3), dims=4)  
    JJ_zs = J .- mean(J, dims = 3) .- mean(J, dims = 4) .+ J0 
    @tullio e[a,b] := JJ_zs[i,j,a,b]*(j!=i)
    return (e.+e')./2
end


function get_filt_mat_mf2(K,Vn, filestructs; mindist::Int=6, ppv_cutoff=0.8, order_Martin = [1, 2, 5, 8, 10, 11, 18, 19, 20, 13, 7, 9, 15, 3, 4, 12, 14, 16, 17, 6])
    V = Vn[order_Martin, :]
    Nf = length(K)
    NN = [size(K[f], 1) for f in 1:Nf]
    N = maximum([size(K[f], 1) for f in 1:Nf])
    H = size(K[1], 3)
    KK = zeros(N,N,H)
    counts = zeros(N, N, H)
    for f in 1:Nf
        gr = sel_good_res(K[f], V, filestructs[f], mindist = mindist, ppv_cutoff = ppv_cutoff) 
        println(size(gr))
        for n in 1:size(gr,1)
            for h in 1:H
                KK[gr[n][1], gr[n][2], h] += K[f][gr[n][1], gr[n][2], h]
                counts[gr[n][1], gr[n][2], h] += 1
            end
        end
    end
    
    KK[KK.!=0] ./= counts[KK.!=0]
    
    @tullio res[h] := KK[i,j,h]
    return res
end

function get_filt_mat_mf(K,Vn, filestructs; mindist::Int=6, ppv_cutoff=0.8, order_Martin = [1, 2, 5, 8, 10, 11, 18, 19, 20, 13, 7, 9, 15, 3, 4, 12, 14, 16, 17, 6, 21])
    V = Vn[order_Martin, :]
    Nf = length(K)
    NN = [size(K[f], 1) for f in 1:Nf]
    N = maximum([size(K[f], 1) for f in 1:Nf])
    H = size(K[1], 3)
    KK = zeros(N,N,H)
    counts = zeros(N, N, H)
    for f in 1:Nf
        gr = sel_good_res(K[f], V, filestructs[f], mindist = mindist, ppv_cutoff = ppv_cutoff) 
        println(size(gr))
        for n in 1:size(gr,1)
            for h in 1:H
                KK[gr[n][1], gr[n][2], h] += K[f][gr[n][1], gr[n][2], h]
                counts[gr[n][1], gr[n][2], h] += 1
            end
        end
    end
    
    KK[KK.!=0] ./= counts[KK.!=0]
    @tullio J[i,j,a,b] := KK[i,j,h]*V[a,h]*V[b,h]
    J0 = mean(mean(J, dims = 3), dims=4)  
    JJ_zs = J .- mean(J, dims = 3) .- mean(J, dims = 4) .+ J0 
    @tullio e[a,b] := JJ_zs[i,j,a,b]*(j!=i)
    e_tot = (e .+ e')./2 
    return e_tot
end
  

function compute_referencescore(score,dist::Dict; mindist::Int=6, cutoff::Number=8.0)
    nc2 = length(score)
    #nc2 == size(d,1) || throw(DimensionMismatch("incompatible length $nc2 $(size(d,1))"))
    out = Tuple{Int,Int,Float64,Float64, Int64}[]
    ctrtot = 0
    ctr = 0
    contact = 0
    for i in 1:nc2
        sitei,sitej,plmscore = score[i][1],score[i][2], score[i][3]
        dij = if haskey(dist,(sitei,sitej)) 
            dist[(sitei,sitej)]
        else
           continue
        end
        if sitej - sitei >= mindist 
            ctrtot += 1
            if dij < cutoff
                ctr += 1
                push!(out,(sitei,sitej, plmscore, ctr/ctrtot, 1))
            else
                push!(out,(sitei,sitej, plmscore, ctr/ctrtot, 0))
            end
        end
    end 
    out
end

function score(K, V; min_separation::Int=6)

    L, L, H = size(K)
    q, H = size(V)
    @tullio Jtens[a, b, i, j] := K[i, j, h] * (j != i) * V[a, h] * V[b, h]

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))


    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end

function score_full(K, V; min_separation::Int=6)

    L, H = size(K)
    q, H = size(V)
    @tullio Jtens[a, b, i, j] := K[i, h] * K[j, h] * V[a, h] * V[b, h] * (j != i)

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))


    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end

function mean_top_cont(KK;top_c = 20)
    M = deepcopy(KK)
    N = size(M,1)
    T = zeros(N,N)
    count = zeros(N,N)

    for h in 1:size(M,3)
        for n in 1:top_c
            i,j = convert(Tuple,argmax(abs.(M[:,:,h])))
            T[i,j] += M[i,j,h] #* VV[h]
            count[i,j] += 1
            M[i,j,h]=0
        end
    end
    #T[T.!=0] ./= count[count .!=0]
    return T
end

function score_full2(K, V; min_separation::Int=6, top_c = 20)

    L, H = size(K)
    q, H = size(V)
    @tullio KK[i, j, h] := K[i, h] * K[j, h] *(j != i)
    #@tullio VVV[a,b,h] := V[a,h] * V[b,h]
    #VV = dropdims(dropdims(mean(mean(abs.(VVV), dims=1), dims=2), dims=1), dims=1)
    #println(size(VV))

    KK = 0.5 * (KK .+ permutedims(KK,[2,1,3]))
    KKK = zeros(L,L,H)
    for h in 1:H
        KKK[:,:,h] = correct_APC(KK[:,:,h])
    end
    
    #KK = KK .- mean(KK, dims=1) .- mean(KK, dims=2) .+ mean(mean(KK,dims=1), dims=2)
    FN = mean_top_cont(KK, top_c = top_c)
    #FN = FN .- mean(FN, dims=1) .- mean(FN, dims=2) .+ mean(mean(FN,dims=1), dims=2)
    FNapc = correct_APC(FN)
    return compute_ranking(abs.(FN), min_separation)
end

function score(Jtens; min_separation::Int=6)
    q,q,L,L = size(Jtens)
    
    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))

    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
end


function compute_fn(J::AbstractArray{T,4}) where {T<:AbstractFloat}
    q, q, L, L = size(J)
    fn = zeros(T, L, L)
    for i in 1:L
        for j in 1:L
            s = zero(T)
            for a in 1:q-1, b in 1:q-1
                s += J[a, b, i, j]^2
            end
            fn[i, j] = s
        end
    end
    # return fn
    return (fn + fn') * T(0.5)
end

function correct_APC(S::Matrix)
    N = size(S, 1)
    Si = sum(S, dims=1)
    Sj = sum(S, dims=2)
    Sa = sum(S) * (1 - 1 / N)
    S -= (Sj * Si) / Sa
    return S
end

function compute_ranking(S::Matrix{Float64}, min_separation::Int = 6)
    N = size(S, 1)
    R = Array{Tuple{Int,Int,Float64}}(undef, div((N-min_separation)*(N-min_separation+1), 2))
    counter = 0
    for i = 1:N-min_separation, j = i+min_separation:N
        counter += 1
        R[counter] = (i, j, S[j,i])
    end

    sort!(R, by=x->x[3], rev=true)
    return R 
end



function compute_PPV(score::Vector{Tuple{Int,Int,Float64}}, filestruct::String; min_separation::Int = 6)
    dist = compute_residue_pair_dist(filestruct)
    return map(x->x[4], compute_referencescore(score, dist, mindist = min_separation))
end

function compute_actualPPV(filestruct;cutoff=8.0,min_separation=6)
    distances=readdlm(filestruct)
    L,_ = size(distances)
    l = 0
    trivial_contacts = 0
    for i in 1:L
        if distances[i,end]<cutoff #originally it was [i,4]
            if abs(distances[i,1]-distances[i,2]) > min_separation
                l += 1
            else 
                trivial_contacts += 1
            end
        end
    end
    println("l = $l")
    println("L = $L")
    println("trivial contacts = $trivial_contacts")
    x = zeros(l)
    fill!(x,1.0)
    scra = map(x->l/x,[l+1:(L-trivial_contacts);])
    return vcat(x,scra) 
    
end
