function compute_residue_pair_dist(filedist::String)
    d = readdlm(filedist)
    if size(d,2) == 4 
        return Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,4] for i in 1:size(d,1) if d[i,4] != 0)
    elseif size(d,2) == 3
        Dict((round(Int,d[i,1]),round(Int,d[i,2])) => d[i,3] for i in 1:size(d,1) if d[i,3] != 0)
    end

end

function compute_referencescore(score,dist::Dict; mindist::Int=6, cutoff::Number=8.0)
    nc2 = length(score)
    #nc2 == size(d,1) || throw(DimensionMismatch("incompatible length $nc2 $(size(d,1))"))
    out = Tuple{Int,Int,Float64,Float64}[]
    ctrtot = 0
    ctr = 0
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
            end
            push!(out,(sitei,sitej, plmscore, ctr/ctrtot))
        end
    end 
    out
end

function score(K, V; min_separation::Int=6)

    L, L, q = size(K)
    q, H = size(V)
    @tullio Jtens[a, b, i, j] := K[i, j, h] * (j != i) * V[a, h] * V[b, h]

    Jt = 0.5 * (Jtens + permutedims(Jtens,[2,1,4,3]))


    ht = zeros(eltype(Jt), q, L)
    Jzsg, _ = gauge(Jt, ht, ZeroSumGauge())
    FN = compute_fn(Jzsg)
    FNapc = correct_APC(FN)
    return compute_ranking(FNapc, min_separation)
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
