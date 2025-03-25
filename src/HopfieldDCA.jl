module HopfieldDCA

import Printf:@printf
import Statistics:mean
using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles
using Optim
using Statistics

import Flux
import Flux: DataLoader, Adam, gradient
import Flux.Optimise: update! 
import Flux.Optimisers: setup
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
import StatsBase: sample, ProbabilityWeights
using LoopVectorization
using LogExpFunctions
using Tullio
using JLD2

include("types.jl")
include("model_from_corr.jl")
include("utils.jl")
include("loss_grad.jl")
include("dca_score.jl")
include("trainers.jl")
include("multitrainer.jl")
include("generative.jl")
include("hopf_to_full.jl")


export quickread, HopPlmVar_gen, HopPlmVar_full, folders_dict, seq_paths_dict, structs_dict, score_hopf
export get_loss_J, trainer, trainer_J, trainer_fullJ, trainer_J_fixedV, get_loss_fullJ, get_loss_hop, trainer_hop 
export multitrainer, multitrainer_full, multitrainer_fixedV, score, compute_PPV, ar_gen, trainer_J_givenV
end
