# Estimate the c value and d value in algorithm 3 in the appendix
# on cliff walking(CW) domain

include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots

T =10000
mdp_dir = "experiment/domain/MDP/"
domains = readdir(mdp_dir)
#cliff walking (CW) domain
d= "cliff.jld2"

domain = d[1:end-5]
mdp = load_jld(mdp_dir * d)["MDP"]

# uniform distribution over all non-sink states
u_s0 = ones(mdp.lSl) ./ (mdp.lSl-1)
u_s0[end] = 0 # Sink state
B = [1e-10] # βc = 1e-10

for β in B
    βs = [β] # β value
    obj = Objective( ρ="ERM", T=T, pars=βs, parEval=βs)

    # -----------------------------------------------------
    # |                 ERM Q-learning                    |
    # -----------------------------------------------------
    function erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
        target = R .+ (γ .* maximum(q_s_,dims=2))[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
        gradient = (exp.(-β .* (target .- q_sa)) .- 1)
        q_sa .-= η .* gradient # (lQl)
        return maximum(q_sa)
    end
    lr_settings = Dict()
    lr_settings["erm"] = Dict("loss"=>erm_loss)
    n_steps = 10000
    eval_every = Int(n_steps/10)
    seed = 0

    # initialize Q learning parameter
    Q_obj = Objective(ρ="ERM", pars=βs, parEval=βs,T=-1)
    # Create a vector (S x A) of all possible s,a pair 
    states = repeat(mdp.S, inner=[mdp.lAl]) # (Batch)
    actions = repeat(mdp.A, outer=[mdp.lSl]) # (Batch)
    # Handle invalid actions  
    invalid = [(sum(r) == -Inf) for r in view.(Ref(mdp.R),states,actions,:)]
    invalid_states = states[invalid]
    invalid_actions = actions[invalid]
    states = states[.!invalid]
    actions = actions[.!invalid]
    total_valid = length(states)
    # Sample transitions for each valid (s,a) pair
    states_,rewards = 0,0
    # -----------------------------------------------------
    #               Actual Q learning update
    # iterate over n update states, 
    # for each iteration we update all state action pair once.
    # -----------------------------------------------------
    Random.seed!(seed)
    
    lr = Counter(1e-8,harmonicDecay,decay_rate=1e-4,ϵ = 0.1)
    sa_method = "erm"
    loss_fun = lr_settings[sa_method]["loss"]
    q = zeros(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
    for (s,a) in zip(invalid_states,invalid_actions)
        q[s,:, a] .= -Inf
    end
   
    c = 0.0 # c value in algorithm 3
    x = zeros(mdp.lSl,mdp.lAl) # keep track of returns of all state-action pairs
    for i in ProgressBar(1:n_steps)
        if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
            states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
            rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
            GC.gc()
        end
        ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
        # step and update on loss function
        for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
            temp = loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
            # save the 
            if (temp > c)
                c = temp
            end
            x[s,a] = x[s,a] + r # keep track of the return for each state-action pair
        end     
        increase(lr)
    end

    x_max = maximum(x)
    x_min = minimum(x)
    d_estimation = (x_max - x_min)^2/8

    println("The estimated d value is : ", d_estimation)
    println("The estimated c value is : ", c)

end

