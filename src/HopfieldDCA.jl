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
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using Tullio

include("types.jl")
include("utils.jl")
include("loss_grad.jl")
include("dca_score.jl")

export quickread, HopPlmVar_gen, Stg, get_anal_grad, get_loss_parts
export get_loss_J, trainer, trainer_J, SmallStg, get_new_grad

end
