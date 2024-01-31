module HopfieldDCA

import Printf:@printf
import Statistics:mean
using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles

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
include("dca_score.jl")

export quickread
export HopPlmVar, get_loss, get_loss_new, get_loss_zyg_francesco, get_loss_zyg, get_loss_pagnani, trainer, trainer_sf, get_loss_sf, trainer_J, trainer_sf_J

end
