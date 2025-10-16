"""
Risk leve α = 0.2, this is for figure 4
Given different number of samples and their EVaR values, 
plot their mean values and their standard deviations

Those EVaR values are computed by q_cliff_multiple.jl under 
the directory q_learning/code/experiments
"""

using Plots
using Statistics

pgfplotsx()

# For cliff walking domain, α = 0.2
samples = [ 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 30000]
evar_seed_0 = [-0.16285,0.222107,0.56861,0.8807867,1.0514988,1.29014,1.31532,1.32109,1.32159]
evar_seed_100 =  [-0.161839,0.207432,0.5943626,0.885596,1.0077894,1.27878,1.316078,1.3219,1.323098]
evar_seed_444 = [-0.181567,0.242399,0.5259503,0.836298,1.069646,1.276831,1.31544,1.319876,1.3218]
evar_seed_5000 =[-0.178342,0.23088,0.537556,0.9031598,1.079787,1.291447,1.3129559,1.320710,1.32123]
evar_seed_10000 =[-0.18003,0.1762558,0.57749478,0.85704153,1.08606925,1.2839104,1.31508,1.320326,1.3212589]
evar_seed_22222 = [-0.16859, 0.22367,0.5485936,0.8856425,1.125295727,1.2540930,1.31303536,1.3200024,1.32233]

# calculate mean and standard deviation across seeds
means = zeros(length(samples))
stds = zeros(length(samples))

for i in 1:length(samples)
  means[i] = mean([evar_seed_0[i], evar_seed_100[i], evar_seed_444[i], evar_seed_5000[i], evar_seed_10000[i], evar_seed_22222[i]])
  stds[i] = std([evar_seed_0[i], evar_seed_100[i], evar_seed_444[i], evar_seed_5000[i], evar_seed_10000[i], evar_seed_22222[i]])
end

plt = plot(
  samples,
  means,
  ribbon=stds,
  label="",
  xlabel="number of samples",
  ylabel="EVaR value",
  xticks=10000: 10000:30000,
  title="",
  legend=:bottomright,
  grid=true,
  size=(340,240)
)
savefig(plt, "mean-std-evars.pdf")