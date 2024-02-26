function norma_col(inp)
    x = copy(inp)
    for (i,col) in enumerate(eachcol(x))
        x[:,i] = inp[:,i] ./ norm(col)
    end
    return x
end

function orthotullio(V)
    n_V = V ./ sqrt.(sum(abs2, V, dims=1))
    @tullio cosin[h1,h2] := n_V[a, h1] * n_V[a, h2]
    return (sum(abs,cosin)-size(V,2))/2
end

function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

softmax(x::AbstractArray{T}; dims = 1) where {T} = softmax!(similar(x, float(T)), x; dims)

softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out .= exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    out ./= sum(out; dims)
end

function softmax_notinplace(x::AbstractArray; dims = 1)
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out = exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    return out ./ sum(out; dims)
end

function quickread(fastafile; moreinfo=false)  
    Weights, Z, N, M, _ = ReadFasta(fastafile, 0.9, :auto, true, verbose = false);
    moreinfo && return Weights, Z, N, M
    return Matrix{Int8}(Z), Weights
end

function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool;verbose=true)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z,verbose=verbose)
    end
    N, M = size(Z)
    q = round(Int,maximum(Z))
    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta,verbose=verbose)
    println("Meff = $(Meff)")
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W,Zint,N,M,q
end


order = [14, 35, 72, 76, 169, 595, 677, 763, 13354]
folders = ["../DataAttentionDCA/data/PF00014/",
    "../DataAttentionDCA/data/PF00035/",
    "../DataAttentionDCA/data/PF00072/",
    "../DataAttentionDCA/data/PF00076/",
    "../DataAttentionDCA/data/PF00169/",
    "../DataAttentionDCA/data/PF00595/",
    "../DataAttentionDCA/data/PF00677/",
    "../DataAttentionDCA/data/PF00763/",
    "../DataAttentionDCA/data/PF13354/"]
seq_paths = ["../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00035/PF00035_full.fasta",
    "../DataAttentionDCA/data/PF00072/PF00072_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00076/PF00076_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00169/PF00169_full.fasta",
    "../DataAttentionDCA/data/PF00595/PF00595_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00677/PF00677_full.fasta",
    "../DataAttentionDCA/data/PF00763/PF00763_full.fasta",
    "../DataAttentionDCA/data/PF13354/PF13354_wo_ref_seqs.fasta.gz"]
structs = ["../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    "../DataAttentionDCA/data/PF00035/Atomic_distances_PF00035.dat",
    "../DataAttentionDCA/data/PF00072/PF00072_struct.dat",
    "../DataAttentionDCA/data/PF00076/PF00076_struct.dat",
    "../DataAttentionDCA/data/PF00169/Atomic_distances_PF00169.dat",
    "../DataAttentionDCA/data/PF00595/PF00595_struct.dat",
    "../DataAttentionDCA/data/PF00677/Atomic_distances_PF00677.dat",
    "../DataAttentionDCA/data/PF00763/Atomic_distances_PF00763.dat",
    "../DataAttentionDCA/data/PF13354/PF13354_struct.dat"]

# Create dictionaries
folders_dict = Dict(zip(order, folders))
seq_paths_dict = Dict(zip(order, seq_paths))
structs_dict = Dict(zip(order, structs))
