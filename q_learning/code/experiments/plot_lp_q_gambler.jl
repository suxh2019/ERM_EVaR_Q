# Plot the difference in EVaR values computed by Linear programming and Q-learning on gambler ruin(GR) domain
# EVaR risk level α = 0.2
# This is for figure 6

include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots
using CSV,DataFrames

# Run q_gambler_0.4.jl. Each number in samples represent the number of samples 
samples = [20,200,1000,3000,5000,10000,15000,20000,30000]

#EVaR values computed by running q_gambler_0.4.jl for different numbers of samples above
evar =[-0.18386,0.3488806,1.0868018,1.100328,1.100328,1.100328,1.10032898,1.10032898,1.100328]

#EVaR values computed by linear program in the lp_gambler.jl under directory linear_program
evar_lp = 1.100524


pgfplotsx()

diff_lp_q =[]

for i in evar
    push!(diff_lp_q,evar_lp -i)  
     
end
  p = plot(samples,diff_lp_q,size=(330,240),label ="", color = "blue",legend=:topright,xticks=10000: 10000:30000)
    xlabel!("number of samples")
    ylabel!("EVaR value difference")
    savefig(p, "evar_difference_gambler.pdf")

