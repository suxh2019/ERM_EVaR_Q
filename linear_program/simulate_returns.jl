"""
plot the distribution of returns of EVaR optimal policies on cliff walking(CW) domain.
This is for figure 3 in the paper.
EVaR risk levels are α = 0.2 and α =0.6 
The EVaR optimal policies are computed by q_cliff_0.2. jl and q_cliff_0.6.jl, which
are under the directory q_learning/code/experiments

"""

using MDPs
import Base
using Revise
using RiskMDPs
using Statistics
using Distributions
using Accessors
using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using CSV: File
using Infiltrator
using CSV
using Plots
using Distributions

pgfplotsx()


"""
load a transient mdp from a csv file, 1-based index
"""
function load_mdp(input)
    mdp = DataFrame(input)
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)
    
    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{IntState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{IntAction}(undef, maximum(sd.idaction))
       
        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            try 
            actions[idaction] = IntAction(ad.idstateto, ad.probability, ad.reward)
            catch e
                error("Error in state $(idstate-1), action $(idaction-1): $e")
            end
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                " that were uninitialized " * string(findall(.!action_init) .- 1 ) ))

        states[idstate] = IntState(actions)
        state_init[idstate] = true
    end
    IntMDP(states)
end


####

# evaluates the policy by simulation
function evaluate_policy(model::TabMDP, π::Vector{Int})

    # evaluation helper variables
    episodes = 1000
    horizon::Integer = 20000
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    states_number = state_count(model)
    returns = []
    
    for inistate in 1: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)

        # rets is a vector of the total rewards, size of rets = number of episodes
        rets = rweights' * H.rewards |> vec
            
        # returns is an array of returns for all episodes
        for r in rets
            push!(returns,r) 
        end
    end  
    returns
end


function plot_histogram(returns1,returns2)
    d1 = Dict()
    for i in returns1
       if i in keys(d1)
           d1[i] += 1
       else
           d1[i] = 1
       end
    end
    d1 = sort(d1,by=first)
    rewards1 = []
     probs1=[]
    for (i,j) in d1
       d1[i] = j /length(returns1)
       push!(rewards1, i)
       push!(probs1, d1[i])
    end

    d2 = Dict()
    for i in returns2
       if i in keys(d2)
           d2[i] += 1
       else
           d2[i] = 1
       end
    end
    d2 = sort(d2,by=first)
    rewards2 = []
     probs2=[]
    for (i,j) in d2
       d2[i] = j /length(returns2)
       push!(rewards2, i)
       push!(probs2, d2[i])
    end

    p= scatter(rewards1,probs1,markersize = 3,size =(350,240), label ="α = 0.2")
    scatter!(rewards2,probs2,markersize = 3,label ="α = 0.6")

    plot!(rewards1,probs1,markersize = 3, linestyle = :dash, linecolor= p.series_list[1][:fillcolor], label = nothing)
    plot!(rewards2,probs2,markersize = 3, linestyle = :dash,linecolor= p.series_list[2][:fillcolor],label = nothing)

    
    xlabel!("return")
    ylabel!("probability")
    savefig(p, "histogram.pdf")
end

function main()

    #EVaR optimal policy for risk level of EVaR: α1 = 0.2
    π1 = [4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1,1, 2, 1, 1, 1, 1, 1, 1, 1]
   
    #EVaR optimal policy for risk level of EVaR:α2 = 0.6
    π2 = [3, 3, 3, 3, 3, 4, 4, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  
    filepath = joinpath(dirname(pwd()), "linear_program","cliff.csv")
    model = load_mdp(File(filepath))

    # compute returns for four policies and save the distribution of
    # the final capitals
    println("------simulate returns----------")
     returns1 = evaluate_policy(model, π1)
     returns2 = evaluate_policy(model, π2)
     
    
     println("---------plot return distribution-----")
     # Sort the returns
     returns1 = sort(returns1)
     returns2 = sort(returns2)
     std1= std(returns1) # standard deviation for α = 0.2
     std2 = std(returns2) # standard deviation for α = 0.6
     mean1 = mean(returns1) # mean value for α = 0.2
     mean2 = mean(returns2) # mean value for α = 0.6
     println("alpha =0.2, mean is ", mean1)
     println("alpha =0.6, mean is ", mean2)
     println("alpha =0.2, standard deviation is ", std1)
     println("alpha =0.6, standard deviation is ", std2)

     plot_histogram(returns1, returns2)   
     
end

main()
