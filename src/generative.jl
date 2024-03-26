


function cond_probs(seq, K, V, h; q=21)
    N = size(K,1)
    res = zeros(size(seq))
    @tullio J[a, i, b, j] := K[i, j, h] * V[a, h] * V[b, h]
    en1 = h[:,1]
    log_z1 = dropdims(LogExpFunctions.logsumexp(en1, dims=1), dims=1)
    p1 = exp.(en1 .- log_z1)
    res[1] = p1[seq[1]]
    for i in 2:N
        en0 = zeros(21)
        for a in 1:21
            for j in 1:i-1
                en0[a]= J[a, i, seq[j], j] 
            end
        end
        en = en0 .+ h[:,i]
        log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
        p = exp.(en .- log_z)
        res[i] = p[seq[i]] 
    end
    return res    
end

function ar_gen(K, V, h)
    N = size(K,1)
    seq = []#Int8.(ones(N))
    @tullio J[a, i, b, j] := K[i, j, h] * V[a, h] * V[b, h]
    
    en1 = h[:,1]
    log_z1 = dropdims(LogExpFunctions.logsumexp(en1, dims=1), dims=1)
    p1 = exp.(en1 .- log_z1)
    #println(sum(p1))
    push!(seq, Int8.(sample(1:21, ProbabilityWeights(p1))))
    #println(seq[1])
    
    for i in 2:N
        en0 = zeros(21)
        for a in 1:21
            for j in 1:i-1
                en0[a]= J[a, i, seq[j], j] 
            end
        end
        en = en0 .+ h[:,i]
        log_z = dropdims(LogExpFunctions.logsumexp(en, dims=1), dims=1)
        p = exp.(en .- log_z)
        push!(seq, Int8.(sample(1:21, ProbabilityWeights(p))))
    end
    
    return seq
end

function ar_gen(K, V, h, n_seq::Int)
    msa = zeros(size(K,1), n_seq)
    for i in 1:n_seq
        msa[:,i] = ar_gen(K, V, h)
    end
    return Int8.(msa)
end


function ar_gen(K, V)
    N = size(K,1)
    seq = []#Int8.(ones(N))
    push!(seq, Int8.(rand(1:21)))
    
    for i in 2:N
        en0 = zeros(21)
        for a in 1:21
            for j in 1:i-1
                en0[a]= J[a, i, seq[j], j] 
            end
        end
        log_z = dropdims(LogExpFunctions.logsumexp(en0, dims=1), dims=1)
        p = exp.(en0 .- log_z)
        push!(seq, Int8.(sample(1:21, ProbabilityWeights(p))))
    end
    
    return seq
end

function ar_gen(K, V, n_seq::Int)
    msa = zeros(size(K,1), n_seq)
    for i in 1:n_seq
        msa[:,i] = ar_gen(K, V)
    end
    return Int8.(msa)
end
