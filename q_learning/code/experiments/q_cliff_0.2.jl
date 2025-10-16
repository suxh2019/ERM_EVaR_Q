# Using algorithm 2 to Compute the optimal EVaR policy 
# on cliff walking(CW) domain
# the optimal policy is for figure 1 and figure 3
# EVaR risk level α = 0.2


include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots

T=10000
mdp_dir = "experiment/domain/MDP/"
filename = "experiment/run/train/out_$(T).jld2"
testfile = "experiment/run/test/evals_$(T).jld2"
domains = readdir(mdp_dir)

d= "cliff.jld2"
domain = d[1:end-5]
mdp = load_jld(mdp_dir * d)["MDP"]

# Compute the set B, the set of β values
function compute_B(α,δ,beta0)
    c = log(1.0/α)
    βk = c / δ  # βk
    B = []
    push!(B,beta0)
    # The actual optimal β value is pretty small
    # give large enough β bound to compute the optimal policy
    # z_{\min} and z_{max} has similar effect
    maxBetaValue = 10
    
    beta = beta0
    while beta < βk && beta <  maxBetaValue 
        beta_next = beta * c /(c-beta*δ) 
        if beta_next < βk
            push!(B,beta_next)
        end
        beta = beta_next
    end
    push!(B,βk)

    return B
end

function compute_h(v,s0,βs)
    sum_exp = 0.0
    for index in 1:length(v)
       sum_exp += s0[index] * exp(-βs[1] * v[index])
    end

    result = -inv(βs[1]) * log(sum_exp)
    return result +log(α)/βs[1]
end


# uniform distribution over all non-sink states
u_s0 = ones(mdp.lSl) ./ (mdp.lSl-1)
u_s0[end] = 0 # Sink state

α = 0.2
δ = 0.1
beta0 = 0.1
B = compute_B(α,δ,beta0)

evar_value = -Inf
optimal_policy = []

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
    end
    lr_settings = Dict()
    lr_settings["erm"] = Dict("loss"=>erm_loss)
    # Multi threaded
    n_threads = Threads.nthreads()
    n_steps = 30000
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
    lr = Counter(1e-10,harmonicDecay,decay_rate=1e-4,ϵ = 0.001)
    sa_method = "erm"
    loss_fun = lr_settings[sa_method]["loss"]
    q = zeros(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
    for (s,a) in zip(invalid_states,invalid_actions)
        q[s,:, a] .= -Inf
    end

    for i in ProgressBar(1:n_steps)
        if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
            states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
            rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
            GC.gc()
        end
        ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
        # step and update on loss function
        for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
            loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
        end     
        increase(lr)
    end
    soln = mdp_out(apply(maximum,q,dims=3),apply(argmax,q,dims=3),obj.pars)
    q_v1 = soln["v"][:,1]
    q_pi1 = soln["π"][:,1]
    h = compute_h(q_v1,u_s0,βs)
    if isnan(h )
        break
    end
    if (h >= evar_value)
        global evar_value = h
        global optimal_policy = deepcopy(q_pi1)
    end
end

println("EVaR risk level α = 0.2, the optimal policy on CW domain is : ", optimal_policy)



