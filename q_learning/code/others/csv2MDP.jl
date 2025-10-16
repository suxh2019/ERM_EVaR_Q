include("../TabMDP.jl")
include("../utils.jl")
using DataFrames
using CSV

csv_dir = "experiment/domain/csv/"
mdp_dir = "experiment/domain/MDP/"
info = CSV.read("experiment/domain/domains_info.csv", DataFrame)
domains = readdir(csv_dir)

for d in domains
    df = CSV.read(csv_dir*d, DataFrame)
    row = (info.domain .== d)
    mdp = df2MDP(df,info.discount[row][1])
    data = Dict("MDP" => mdp)
    save_jld(mdp_dir*d[1:end-4]*".jld2",data)
end
