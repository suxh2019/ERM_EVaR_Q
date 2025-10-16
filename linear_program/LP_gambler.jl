"""
Calculate EVaR values computed by linear program on gambler ruin(GR) domain
Risk leve α = 0.2, this is for figure 6
"""

using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using MDPs
using JuMP, HiGHS
using CSV: File
using RiskMDPs
using Plots
using Infiltrator
using CSV

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

"""
Discretize β for EVaR
"""
function evar_discretize_beta(α::Real, δ::Real)
    zero(α) < α < one(α) || error("α must be in (0,1)")
    zero(δ) < δ  || error("δ must be > 0")

    # set the smallest and largest values
    β1 = 2e-7
    βK = -log(α) / δ

    βs = Vector{Float64}([])
    β = β1

    # experimenting on small β values
    β_bound = minimum([10,βK]) 
    while β < β_bound
        append!(βs, β)
        β *= log(α) / (β*δ + log(α)) *1.05

    end

    βs

end

"""
Compute B[s,s',a],  b_s^d, B_{s,s'}^d, d_a(s), assume the decsion rule d is deterministic,that is,
 d_a(s) is always 1. 
 a is the action taken in state s
when sn is the sink state, then B[s,a,sn] =  b_s^d, 
when sn is a non-sink state,   B[s,a,sn] = B_{s,s'}^d.
"""
function compute_B(model::TabMDP,β::Real)
    
    states_size = state_count(model)
    actions_size = maximum([action_count(model,s) for s in 1:states_size])

    B = zeros(Float64 , states_size, actions_size,states_size)
     
    for s in 1: states_size
        action_number = action_count(model,s)
        for a in 1: action_number
            snext = transition(model,s,a)
            for (sn, p, r) in snext
                B[s,a,sn] += p  * exp(-β * r) 
            end
        end
    end 
    B
end

"""
Linear program to compute erm exponential value function w and the optimal policy
Assume that the last state is the sink state
β is the risk level of ERM
"""
function erm_linear_program(model::TabMDP, B::Array, β::Real)
    
     state_number = state_count(model)
     # policy π 
     π = zeros(Int , state_number)
     constraints::Vector{Vector{ConstraintRef}} = []

     lpm = Model(HiGHS.Optimizer)
     set_silent(lpm)
     @variable(lpm, w[1:(state_number-1)] )
     @objective(lpm, Min, sum(w))

    #constraints for non-sink states and all available actions
    for s in 1: state_number-1
        action_number = action_count(model,s)
        c_s::Vector{ConstraintRef} = []
        for a in 1: action_number
            push!(c_s, @constraint(lpm, w[s] ≥ B[s,a,1:(state_number-1)] ⋅ w -
                      B[s,a,state_number] ))
        end
        push!(constraints, c_s)
    end

    optimize!(lpm)

    # Check if the linear program has a feasible solution 
    if termination_status(lpm) ==  DUAL_INFEASIBLE || w[(state_number - 1) ] == 0.0
        return  (status = "infeasible", w=zeros(state_number),v=zeros(state_number),π=zeros(Int64,state_number))
    else
         # Exponential value functions
         w = vcat(value.(w), [-1.0])

         #Regular value functions 
         v = -inv(β) * log.(-value.(w) )

         # Check active constraints to obtain the optimal policy
         π = vcat(map(x->argmax(dual.(x)), constraints), [1])
       
        return (status ="feasible", w=w,v=v,π=π)
    end 
end

# Compute a single ERM value using the vector of regular value function and initial distribution
function compute_erm(value_function :: Vector, initial_state_pro :: Vector, β::Real)

    # -1.0/β * log(∑ μ_s * exp()), g_t(π,β) in Corollary 3.2
    sum_exp = 0.0
    for index in 1:length(value_function)
        sum_exp += initial_state_pro[index] * exp(-β * value_function[index])
    end

    result = -inv(β) * log(sum_exp)

    result
end


# Compute EVaR values for different risk level α 
function compute_optimal_policy(alpha_array,initial_state_pro, model,δ)
    
    #Save erm values and beta values for plotting unbounded erm 
    erm_values = []
    beta_values =[]
    evars = []

    #α: risk level of EVaR 
    for α in alpha_array
        βs =  evar_discretize_beta(α, δ)
        max_h =-Inf

        for β in βs
            B = compute_B(model,β)
            status,w,v,π = erm_linear_program(model,B,β)
            
            # compute the feasible and optimal solution
            if cmp(status,"infeasible") ==0 
                break
            end

            # Calculate erm value. 
            erm = compute_erm(v,initial_state_pro, β)

            # Save erm values and β values for plots
            append!(erm_values,erm)
            append!(beta_values,β)

            # Compute h(β) 
            h = erm + log(α)/β
            if (h == Inf)
                println("overflow ", erm)
            end

            if h  > max_h
                max_h = h
        
            end
        end
        push!(evars,max_h)

    end
    evars
end


function main_evar()

    δ = 0.01
    filepath = joinpath( dirname(pwd()), "linear_program", "gambler.csv")                
    model = load_mdp(File(filepath))
   
    state_number = state_count(model)

    # The intial distribution over states
    initial_state_pro = Vector{Float64}()
    # The first state is a sink state,the initial probability is 0

    append!(initial_state_pro,0)  
    for index in 2:(state_number-1)
        append!(initial_state_pro,1.0/(state_number-2)) 
    end
    # the last state is a sink state, the initial probability is 0
    append!(initial_state_pro,0) 

    # EVaR risk level on gambler ruin(GR) domain
    alpha_array = [0.2]
    
    #Compute the EVaR value and return it
    evars = compute_optimal_policy(alpha_array, initial_state_pro, model,δ)
   
    println("EVaR risk level α is 0.2 on gambler ruin(GR) domain is : ", evars[1])
 
end 

main_evar()







