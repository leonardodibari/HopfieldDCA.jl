module HopfieldDCA

import Printf:@printf
import Statistics:mean
using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles
#using NLopt
import Flux
import Flux: DataLoader, Adam, gradient
import Flux.Optimise: update! 
import Flux.Optimisers: setup
import Zygote:gradient
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using Tullio


include("types.jl")
include("utils.jl")
include("loss_grad.jl")
include("plm_hopfield.jl")
include("dca_score.jl")


export HopPlmVar, get_loss_and_grad, get_loss_and_grad_zyg, trainer_small, define_var, check_with_zyg
export create_storage_arr, get_loss_and_grad_zyg2, get_loss_and_grad_zyg3, StgArr, get_loss_and_grad2, trainer

end
