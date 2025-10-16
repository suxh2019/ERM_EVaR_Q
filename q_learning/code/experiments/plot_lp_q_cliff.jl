# plot the difference in EVaR values computed by linear program and Q-learning on cliff walking(CW) domain
# The EVaR risk level α = 0.2
# This is for figure 5
# EVaR values are calulated by running q_cliff_multiple.jl

include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots
using CSV,DataFrames
using Statistics


pgfplotsx()

function plot_lp_q_cliff()

# For cliff walking domain, α = 0.2; Each number in samples represent the number of samples 
samples = [ 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 30000]

#EVaR values computed by running q_cliff_multiple.jl for different numbers of samples above
evar = [-0.16285,0.222107,0.56861,0.8807867,1.0514988,1.29014,1.31532,1.32109,1.32159]

#EVaR values computed by linear program in the lp_cliff.jl under directory linear_program
evar_lp = 1.32375

diff_lp_q =[]

for i in evar
    push!(diff_lp_q,evar_lp -i)  
     
end
  p = plot(samples,diff_lp_q,size=(300,240),label ="", color = "blue",legend=:topright, xticks=10000: 10000:30000)
    xlabel!("number of samples")
    ylabel!("EVaR value difference")
    savefig(p, "evar_difference_cliff.pdf")

end

plot_lp_q_cliff()
