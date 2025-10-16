include("TabMDP.jl")
include("utils.jl")
using Random

"""
This document contains mdp Environment that act as an online environment.
The function input and output is indentical to standard gym, for extension.
"""

mutable struct mdpEnvironments
    n_env::Int
    mdp::MDP
    states::Vector{Int}
    function mdpEnvironments(n_env::Int,mdp::MDP;uniform::Bool=false)
        new(n_env,mdp,fill(0, n_env))
    end
end

# Implement the step function, simulation step given state and action.
# Input : a
# Output : s', r , terminated, trucated
function step(envs::mdpEnvironments, actions::Vector{I}) where I <: Integer
    states_ = sample1_from_transition.(Ref(envs.mdp.P_sample), envs.states, actions)
    out = (states_ , envs.mdp.R[CartesianIndex.(envs.states, actions, states_)])
    envs.states = states_  # update system state
    return out
end

function reset_states(n_env::Int,mdp::MDP,uniform::Bool=false)
    if uniform
        n_env == (mdp.lSl * mdp.lAl) || error("n_env must be lSl x lAl")
        return repeat(mdp.S, inner=[mdp.lAl])
    else
        return rand(Categorical(mdp.s0),n_env)
    end
end

function reset(envs::mdpEnvironments, uniform::Bool=false)    
    envs.states = reset_states(envs.n_env,envs.mdp,uniform)
    return envs.states
end

function reset_instance(envs::mdpEnvironments, i)
    envs.states[i] = rand(Categorical(envs.mdp.s0))
    return envs.states[i]
end

function eval_out(α,ret)
    return Dict("α" => α, "values" => ret)
end

function simulate(VI_out::Any,obj::Objective,mdp::MDP;ENV_NUM = 10000,T = 1000,seed = 0,quant_ϵ=1e-14)
    if obj.ρ_type == "nest" 
        simulate_fun = (obj.T == -1) ? simulateMarkov : simulateMarkov_TimeDep
        if obj.ρ in Set(["mean","E","min","max"])
            return eval_out(obj.parEval,simulate_fun(VI_out["π"],mdp,ENV_NUM=ENV_NUM,T=T,seed=seed))
        else
            return eval_out(obj.parEval,[simulate_fun(out["π"],mdp,ENV_NUM=ENV_NUM,T=T,seed=seed) for out in VI_out])
        end
    elseif obj.ρ_type == "distMarkov" 
        simulate_fun = (obj.T == -1) ? simulateMarkov : simulateMarkov_TimeDep
        return eval_out(obj.parEval,[simulate_fun(out["π"],mdp,ENV_NUM=ENV_NUM,T=T,seed=seed) for out in VI_out]) 
    elseif obj.ρ_type == "ent"
        simulate_fun = simulateMarkov_TimeDep
        return eval_out(obj.parEval,[simulate_fun(out["π"],mdp,ENV_NUM=ENV_NUM,T=T,seed=seed) for out in VI_out])
    elseif obj.ρ_type == "quant"
        v = VI_out["v"]
        if obj.T == -1
            simulate_fun = simulateQuant 
        else
            simulate_fun = simulateQuant_TimeDep
            v = v[1,:,:]
        end
        if obj.ρ == "Chow"
            v = collect(hcat([(CVaR2X(v[s0, :],obj.pars,obj.pdf)) for s0 in mdp.S]...)')
        end
        d0 = initDistribution(mdp,v, obj.pdf)
        τs = VaR(d0,obj.parEval)
        return Dict(
            "option 1" => eval_out( obj.parEval,[simulate_fun(τ,VI_out["v"],VI_out["π"],mdp,ENV_NUM=ENV_NUM,T=T,renew_τ = true,seed=seed,ϵ=quant_ϵ) for τ in τs] )
            #,"option 2" => eval_out( obj.parEval,[simulate_fun(τ,v,VI_out["π"],mdp,ENV_NUM=ENV_NUM,T=T,renew_τ = false,seed=seed,ϵ=quant_ϵ) for τ in τs] )
        )
    elseif obj.ρ_type == "target"
        if obj.T == -1
            error("target value methods cannot handle infinite horizon yet.")
        end
        τs = initTarget(mdp,VI_out["v"], VI_out["Z0"],obj.parEval)
        return eval_out(obj.parEval,[simulatePrimalTimeDep(τ,VI_out["π"],mdp,ENV_NUM = ENV_NUM,T = T,seed=123,digit=VI_out["digit"]) for τ in τs["opt_z"]])
    end
end



function simulateMarkov_TimeDep(π::Array{I},mdp::MDP;ENV_NUM = 10000,T = 1000,seed=123) where I<:Integer
    Random.seed!(seed)
    envs = mdpEnvironments(ENV_NUM,mdp)
    cum_rew = zeros(ENV_NUM)
    states = reset(envs)
    pi_t_max = size(π)[1]
    for t in 1:T
        actions = π[Base.min(t,pi_t_max),states]
        states,r = step(envs,actions)
        cum_rew += (r * mdp.γ^(t-1))
    end
    return cum_rew
end

function simulateMarkov(π::Vector{I},mdp::MDP;ENV_NUM = 10000,T = 1000,seed=123) where I<:Integer
    Random.seed!(seed)
    envs = mdpEnvironments(ENV_NUM,mdp)
    cum_rew = zeros(ENV_NUM)
    states = reset(envs)
    for t in 1:T
        actions = π[states]
        states,r = step(envs,actions)
        cum_rew += (r * mdp.γ^(t-1))
    end
    return cum_rew
end

function simulateQuant(τ0::Float64,v::Array{Float64},π::Array{I},mdp::MDP;ENV_NUM = 10000,T = 1000,renew_τ = true,seed=123,ϵ=1e-16) where I<:Integer
    Random.seed!(seed)
    lQl = size(v)[end] # v (S , Q)
    envs = mdpEnvironments(ENV_NUM,mdp)
    cum_rew = zeros(ENV_NUM)
    states = reset(envs)
    τ = fill(τ0 ,ENV_NUM)
    for t in 1:T
        αᵢ = Base.min.(τ_to_α.(Ref(v),states,τ),lQl)
        if renew_τ # renew_τ = True : the τ get update every iteration
            τ = v[CartesianIndex.(states,αᵢ)]
        end
        actions = π[CartesianIndex.(states,αᵢ)]
        states,r = step(envs,actions)
        # decreasing τ by a factor of ϵ make it slightly smaller to handle the precision error from floating point arithmetic
        τ = (τ .- r) ./ mdp.γ
        τ .-= maximum(abs.(τ)) * ϵ
        cum_rew += (r * mdp.γ^(t-1))
    end
    return cum_rew
end

function simulateQuant_TimeDep(τ0::Float64,v::Array{Float64},π::Array{I},mdp::MDP;ENV_NUM = 10000,T = 1000,renew_τ = true,seed=123,ϵ=1e-16) where I<:Integer
    Random.seed!(seed)
    lQl = size(v)[end] # v (T, S , Q)
    envs = mdpEnvironments(ENV_NUM,mdp)
    cum_rew = zeros(ENV_NUM)
    states = reset(envs)
    τ = fill(τ0 ,ENV_NUM)
    for t in 1:T
        v_cur = @view v[t,:,:]
        π_cur = @view π[t,:,:]
        αᵢ = Base.min.(τ_to_α.(Ref(v_cur),states,τ),lQl)
        if renew_τ # renew_τ = True : the τ get update every iteration
            τ = v_cur[CartesianIndex.(states,αᵢ)]
        end
        actions = π_cur[CartesianIndex.(states,αᵢ)]
        states,r = step(envs,actions)
        # decreasing τ by a factor of ϵ make it slightly smaller to handle the precision error from floating point arithmetic
        τ = (τ .- r)
        τ = (τ .* (ifelse.( τ .< 0 , ((1+ϵ)/mdp.γ) , ((1-ϵ)/mdp.γ))))

        cum_rew += (r * mdp.γ^(t-1))
    end
    return cum_rew
end

# This methods currently only handle CVaR by Baurle (2011)
function simulatePrimalTimeDep(τ0::Float64,policy,mdp::MDP;ENV_NUM = 10000,T = 1000,seed=123,digit=2) 
    Random.seed!(seed)
    envs = mdpEnvironments(ENV_NUM,mdp)
    cum_rew = zeros(ENV_NUM)
    states = reset(envs)
    τ = fill(τ0 ,ENV_NUM)
    lim_z = [(isempty(keys(policy[t,s])) ? nothing : extreme(keys(policy[t,s]))) for t in 1:T, s in mdp.S]

    for t in 1:T
        actions = [policy[t,s][clamp(τ[i],lim_z[t,s]...)] for (i,s) in enumerate(states)]
        states,r = step(envs,actions)
        τ = ceil.( (τ .- r) ./ mdp.γ ,digits=digit)
        cum_rew += (r * mdp.γ^(t-1))
    end
    return cum_rew
end
