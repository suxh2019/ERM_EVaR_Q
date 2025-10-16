include("riskMeasure.jl")
include("utils.jl")
using ProgressBars
using Distributions

# HERE WE DEFINE value function AS A 3D Dictionary
# 1. # of quantiles 2. domain name 3. riskMeasure
# this value function retuns the MDP VI output
function init_jld(filename)
    if isfile( filename )
        return load_jld(filename)
    else
        jld = Dict("1" => Dict())
        save_jld(filename,jld)
        return jld
    end
end

function reset_jld(filename)
    jld = Dict("1" => Dict())
    save_jld(filename,jld)
end


function insert_jld(jld , lQl, domain, ρ, input)
    if !(string(lQl) in keys(jld))
        jld[string(lQl)] = Dict()
    end
    if !(domain in keys(jld[string(lQl)]))
        jld[string(lQl)][domain] = Dict()
    end
    jld[string(lQl)][domain][ρ] = input
end

function in_jld(jld, lQl, domain, ρ)
    if !(string(lQl) in keys(jld))
        return false
    end
    if !(domain in keys(jld[string(lQl)]))
        return false
    end
    if !(ρ in keys(jld[string(lQl)][domain]))
        return false
    end
    return true
end


function mdp_out(v, π, α)
    return Dict("v" => v, "π" => π, "α" => α)
end

viFunDict = Dict(
    "nERM" => Dict( "type" => "nest", "fx" => ERMs , "cdf2pdf" => q_evenPdf),
    "nVaR" => Dict( "type" => "nest", "fx" => VaR , "cdf2pdf" => q_evenPdf),
    "nCVaR" => Dict( "type" => "nest", "fx" => CVaR , "cdf2pdf" => q_evenPdf),
    "nEVaR" => Dict( "type" => "nest", "fx" => EVaR , "cdf2pdf" => q_evenPdf),
    "min" => Dict( "type" => "nest", "fx" => min , "cdf2pdf" => q_evenPdf),
    "max" => Dict( "type" => "nest", "fx" => max , "cdf2pdf" => q_evenPdf),
    "E" => Dict( "type" => "nest", "fx" => E , "cdf2pdf" => q_evenPdf),
    "mean" => Dict( "type" => "nest", "fx" => E , "cdf2pdf" => q_evenPdf),
    "Chow" => Dict( "type" => "quant", "fx" => CVaR , "cdf2pdf" => CVaR_cdf2pdf), # Chow CVaR Value Iteration
    "CVaR" => Dict( "type" => "target", "fx" => E , "cdf2pdf" => q_evenPdf), # Baurle target value CVaR Value Iteration, mean of CVaR utility
    "quant" => Dict( "type" => "quant", "fx" => quant , "cdf2pdf" => q_evenPdf), # Objective.pars := [1/J,2/J,...,J/J]
    "quant_under" => Dict( "type" => "quant", "fx" => quant , "cdf2pdf" => q_evenPdf), # Objective.pars := [0,1/J,...,(J-1)/J]
    "VaR" => Dict( "type" => "quant", "fx" => VaR , "cdf2pdf" => q_evenPdf), # Objective.pars := [0,1/J,...,(J-1)/J]
    "VaR_over" => Dict( "type" => "quant", "fx" => VaR , "cdf2pdf" => q_evenPdf), # Objective.pars := [1/J,2/J,...,J/J]
    "ERM" => Dict( "type" => "ent", "fx" => ERMs , "cdf2pdf" => q_evenPdf),
    "EVaR" => Dict( "type" => "ent", "fx" => EVaR , "cdf2pdf" => q_evenPdf), # Hau EVaR MDP
    "dVaR" => Dict( "type" => "distMarkov", "fx" => VaR , "cdf2pdf" => q_evenPdf ), # VaR_cdf2pdf ) 
    "dCVaR" => Dict( "type" => "distMarkov", "fx" => CVaR , "cdf2pdf" => CVaR_cdf2pdf)
    )


mutable struct Objective
    ρ::String
    T::Int
    δ::Any
    pars::Array
    l::Int
    ρ_type::String
    ρ_fx::Function
    pdf::Array
    parEval::Array # This is particular for DistMarkov type as they could have (discretization , evals)
    function Objective(;ρ::String = "E", T::Int = -1, δ::Any = 0, pars = [1.0],parEval=[1.0])
        ρ ∈ keys(viFunDict) || error(ρ * "-MDP not supported. Please choose from :"*string(allrisks))
        ρfeatures = viFunDict[ρ]
        ρ_type = ρfeatures["type"]
        ρ_fx = ρfeatures["fx"]
        pdf = ρfeatures["cdf2pdf"](pars)
        total_pdf = cumsum(pdf)[end]
        ρ_type == "ent" || ρ_type == "nest" || total_pdf == 1 || error("$ρ pdf must sum to 1, instead sum to $total_pdf")
        new( ρ , T , δ , pars , length(pars) , ρ_type , ρ_fx , pdf ,parEval)
    end
end


struct MDP
    S::Vector{Int}
    lSl::Int
    A::Vector{Int}
    lAl::Int
    R::Array
    P::Array
    γ::Float64
    s0::Array
    # sparse transition matrix and transition states
    S_sa::Array
    P_sa::Array
    # transition sampler
    P_sample::Array
    # valid actions
    valid_A::Vector{Vector{Int}}
end


function VI(mdp::MDP,obj::Objective)
    # Nested algorithms solve recursively using dynamic programming
    if obj.ρ_type == "nest"
        if obj.T == -1  # infinite horizon nested
            out = [ infNestVi(mdp,obj, param=par) for par in obj.pars]
        else  # finite horizon nested
            out =  [ nestVi(mdp,obj, param=par) for par in obj.pars]
        end
        return length(out)==1 ? out[1] : out
    # Target value augmentation approach, is accurate only when target value is sufficiently discretize
    elseif obj.ρ_type == "target"
        if obj.T == -1  # infinite horizon
            error("Infinite horizon for target level MDP is not implemented")
        else  # finite horizon
            return  targetVi(mdp,obj)
        end
    # Quantile augmentation approach, is accurate only when quantile is discretize sufficiently
    elseif obj.ρ_type == "quant"
        if obj.T == -1  # infinite horizon
            return  quantileViInf(mdp,obj)
        else  # finite horizon
            return  quantileVi(mdp,obj)
        end
    # Time augmentation approach, precisely for finite horizon and δ optimal for infinite horizon
    elseif obj.ρ_type == "ent"
        if obj.ρ == "ERM"
            return  ermVi(mdp,obj)
        else obj.ρ == "EVaR"
            return  evarVi(mdp,obj)
        end
    elseif obj.ρ_type == "distMarkov"
        # we run single thread since multi-threaded could overflow RAM and be much slower in big domain
        if obj.T == -1  # infinite horizon
            out = [markovQuantViInf(mdp,obj,par) for par in obj.parEval]
        else  # finite horizon
            out = [markovQuantVi(mdp,obj,par) for par in obj.parEval]
        end
        return length(out)==1 ? out[1] : out
    end
end


function get_ρ_fx(obj::Objective,param::Float64)
    if obj.ρ in Set(["E","mean","min","max"])
        return (X,p) -> obj.ρ_fx(distribution(X,p))
    elseif length(param) == 1
        return (X,p) -> obj.ρ_fx(distribution(X,p),[param])[1]
    end
    error("multiple risk levels passed into nested value iteration")
end

function nestVi(mdp::MDP,obj::Objective; param=1.0)
    v = zeros(obj.T + 1, mdp.lSl)
    q = fill(-Inf,obj.T, mdp.lSl, mdp.lAl)
    # define specific risk measure
    ρᵢ = get_ρ_fx(obj,param)
    # backward induction
    for t in ProgressBar(obj.T:-1:1)
        for s in mdp.S for a in mdp.valid_A[s]
            q[t, s, a] = ρᵢ(mdp.R[s, a, :] .+ mdp.γ .* v[t + 1, :],mdp.P[s, a, :]) 
        end end
        v[t,:] = apply(maximum,q[t,:,:],dims=2)
    end
    π = apply(argmax, q, dims=3)
    return  mdp_out(v, π, param)
end

function infNestVi(mdp::MDP,obj::Objective; param=1.0, ϵ=1e-14, maxiter=500)
    v = zeros(2, mdp.lSl)
    q = fill(-Inf,mdp.lSl, mdp.lAl)
    # define specific risk measure
    ρᵢ = get_ρ_fx(obj,param)
    # backward induction
    t1,t2 = 1,2  # define alternating time variables
    for i in ProgressBar(1:maxiter)
        for s in mdp.S for a in mdp.valid_A[s]
            q[s,a] = ρᵢ(mdp.R[s, a, :] .+ mdp.γ .* v[t2, :],mdp.P[s, a, :]) 
        end end
        v[t1,:] = apply(maximum,q,dims=2)
        if maximum(abs.(v[t1,:] - v[t2,:])) <= ϵ
            π = apply(argmax, q, dims=2)
            return mdp_out(v[t1,:], π, param)
        end
        t1,t2 = t2,t1
    end
    π = apply(argmax, q, dims=2)
    return mdp_out(v[t2,:], π, param)
end

function δ_to_T(mdp::MDP,obj::Objective)
    ΔR = ( maximum(mdp.R) - minimum(mdp.R[(mdp.R .!= -Inf)]) ) / (1 - mdp.γ)
    obj.δ > 0 || error("please set δ > 0, for ERM infinite horizon") 
    max_β = maximum(obj.pars[ obj.pars .> 0])
    return Int(ceil(log(8 * obj.δ / (max_β * (ΔR^2))) / (2 * log(mdp.γ))))
end

# obj.pars can be a list of parameter, param is the single parameter qval is solving.
function ermBackInd(mdp::MDP,obj::Objective, param, term)
    v = zeros(obj.T + 1, mdp.lSl)
    q = fill(-Inf,obj.T, mdp.lSl, mdp.lAl)
    π = zeros(Int,obj.T + 1, mdp.lSl)
    v[obj.T+1, :] .= term["v"]
    π[obj.T+1, :] .= term["π"]
    for t in obj.T:-1:1 
        for s in mdp.S for a in mdp.valid_A[s]
            q[t, s, a] = base_ERM(mdp.R[s, a, :] .+ mdp.γ .* v[t + 1, :],mdp.P[s, a, :],param * mdp.γ ^ t)
        end end
        v[t,:] = apply(maximum,q[t,:,:],dims=2)
    end
    π[1:obj.T,:] = apply(argmax, q, dims=3)
    return  mdp_out(v, π, param)
end

function ermVi(mdp::MDP,obj::Objective)
    if obj.T == -1  # infinite horizon
        # Solve risk neutral Nominal-MDP
        term = infNestVi(mdp,Objective( ρ ="E"))
        if all(obj.pars .== 0)  # all risk level 0 implies mean
            return [term for _ in obj.pars]  # return mean solution
        end
        # generally calculate the time step required to achieve delta error
        obj.T = δ_to_T(mdp,obj)
    else  # finite horizon
        obj.T > 0|| error("please set Objective horizon > 0")
        term = mdp_out(0.0, 0, 0)
    end
    result = Vector{Dict}(undef, length(obj.pars))
    Threads.@threads for (i, param) in ProgressBar(collect(enumerate(obj.pars)))
        result[i] = ermBackInd(mdp,obj,param,term)
    end
    # GC.gc()  # clear up threads information
    return result
end

function evarVi(mdp::MDP,obj::Objective)
    # If α == 1 then is standard MDP, α == 0 min MDP
    if all(obj.pars .== 1)
        soln = VI(mdp,Objective(ρ="E", T=obj.T))
        return [soln for _ in obj.pars]
    elseif all(obj.pars .== 0)
        soln = VI(mdp,Objective(ρ="min", T=obj.T))
        return [soln for _ in obj.pars]
    end
    ΔR = ( maximum(mdp.R) - minimum(mdp.R[(mdp.R .!= -Inf)]) ) / (1 - mdp.γ)

    if obj.T == -1  # If infinite horizon
        obj.δ /= 2  # split half δ for discrete β and approx T each
    else  # If finite horizon
        obj.T > 0 || error("please set Objective horizon > 0")
    end
    α_min = minimum(obj.pars[ obj.pars .> 0])
    βs = 100 * (0.99 .^ (0:3000))

    objERM = Objective( ρ="ERM", T=obj.T, pars=βs, δ=obj.δ)
    ERM_ret = ermVi(mdp,objERM)
    erms_val = [base_ERM(sol["v"][1, :], mdp.s0, β) for (β, sol) in zip(βs, ERM_ret)]
    opt_β = mapslices(argmax, (erms_val' .+ log.(obj.pars) ./ βs') ; dims=2)
    length(opt_β) == length(obj.pars) || "argmax dimension is wrong"

    return [ERM_ret[l] for l in opt_β]
end

# quantile Q return the joint distribution given lSl marginal and conditional distribution
function quantile_q(mdp::MDP,obj::Objective, s::Int, a::Int, V)
    # Imputation strategy
    if obj.ρ == "Chow"
        Xs = [mdp.R[s, a, sn] .+ (mdp.γ .* (CVaR2X(V[sn, :],obj.pars,obj.pdf)) ) for sn in mdp.S_sa[s][a]]
    else 
        Xs = [mdp.R[s, a, sn] .+ (mdp.γ .* (@view V[sn, :]) ) for sn in mdp.S_sa[s][a]]
    end
    # combine the random value and probability, then create a joint distribution (d)
    return jointD_fixPDF(Xs, obj.pdf , mdp.P_sa[s][a])
end

function quantileVi(mdp::MDP,obj::Objective)
    GC.gc()  
    # initialize vectors
    v = zeros(obj.T + 1, mdp.lSl, obj.l)
    q = fill(-Inf,obj.T, mdp.lSl, obj.l, mdp.lAl)
    for t in ProgressBar(obj.T:-1:1)
        for s in mdp.S
            for a in mdp.valid_A[s]
                q[t, s, :, a] = obj.ρ_fx(quantile_q(mdp,obj, s, a, (@view v[t + 1,:,:])),obj.pars)
            end
        end
        v[t,:,:] = apply(maximum,(@view q[t,:,:,:]),dims=3)
    end
    return mdp_out(v, apply(argmax, q, dims=4), obj.pars)
end

function quantileViInf(mdp::MDP,obj::Objective; ϵ::Float64=1e-14, maxiter::Int64=500)
    GC.gc()  
    v = zeros(2, mdp.lSl, obj.l)
    q = fill(-Inf,mdp.lSl, obj.l, mdp.lAl)
    t1,t2 = 1,2  # define alternating time variables
    for i in ProgressBar(1:maxiter)
        for s in mdp.S
            for a in mdp.valid_A[s]
                q[s, :, a] = obj.ρ_fx(quantile_q(mdp,obj, s, a, (@view v[t2,:,:])),obj.pars)
            end
        end
        v[t1,:,:] = apply(maximum,q,dims=3)
        if maximum(abs.(  (@view v[t1,:,:]) - (@view v[t2,:,:]) )) <= ϵ
            π = apply(argmax, q, dims=3)
            return mdp_out(v[t1,:,:], π, obj.pars)
        end
        t1,t2 = t2,t1
    end
    println("Maximum number of iterations reached")
    π = apply(argmax, q, dims=3)
    return mdp_out(v[t2,:,:], π,obj.pars)
end

function initDistribution(mdp::MDP, V::Matrix{Float64}, pdf::Vector{Float64})
    Xs = [V[s0, :] for s0 in mdp.S]
    return jointD_fixPDF(Xs, pdf, mdp.s0)
end

function τ_to_α(v,s,τ)
    return searchsortedfirst((@view v[s,:]),τ)
end

function dist_ρ_fx(obj::Objective,param::Float64)
    if obj.ρ in Set(["E","mean","min","max"])
        return (X) -> obj.ρ_fx(X)
    elseif length(param) == 1
        return (X) -> obj.ρ_fx(X,[param])[1]
    end
    error("multiple risk levels passed into nested value iteration")
end

function markovQuantViInf(mdp::MDP,obj::Objective,param::Float64;ϵ::Float64=1e-14, maxiter::Int64=500)
    GC.gc()  
    # initialize vectors
    v = zeros(2, mdp.lSl, obj.l)
    π = zeros(Int,mdp.lSl)
    ρᵢ = dist_ρ_fx(obj,param) 
    t1,t2 = 1,2  # define alternating time variables
    for t in ProgressBar(1:maxiter)
        for s in mdp.S
            ds = [quantile_q(mdp,obj, s, a, (@view v[t2,:,:])) for a in mdp.valid_A[s]]
            a_opt_index = argmax([ρᵢ(d) for d in ds])
            π[s] = mdp.valid_A[s][a_opt_index]
            v[t1,s,:] = VaR(ds[a_opt_index],obj.pars) 
        end
        t1,t2 = t2,t1
        if maximum(abs.( (@view v[t1,:,:]) .- (@view v[t2,:,:]) )) <= ϵ
            return mdp_out(v[t2,:,:], π, param)
        end
    end
    return mdp_out(v[t2,:,:], π, param)
end

function markovQuantVi(mdp::MDP,obj::Objective,param::Float64)
    # initialize vectors
    v = zeros(obj.T + 1, mdp.lSl, obj.l)
    π = zeros(Int, obj.T, mdp.lSl)
    ρᵢ = dist_ρ_fx(obj,param) 
    for t in ProgressBar(obj.T:-1:1) 
        GC.gc() 
        for s in mdp.S
            ds = [quantile_q(mdp,obj, s, a, (@view v[t + 1,:,:])) for a in mdp.valid_A[s]]
            a_opt_index = argmax([ρᵢ(d) for d in ds])
            π[t, s] = mdp.valid_A[s][a_opt_index]
            v[t,s,:] = VaR(ds[a_opt_index],obj.pars)
        end
    end
    return mdp_out(v, π , param)
end


# A recursive function for target value - CVaR MDP : Average Value at Risk - Baurle (2011)
# Input :
# T : Total time horizon
# v : Value function Matrix of dictionaries v[t,s][z] (value)
# v : Value function Matrix of dictionaries π_[t,s][z] (action)
# t : Current step (start from 1 and end at T)
# s,z : Current state and target level
# mdp : A Markov Decision Process model
# lb,ub : Lower bound and upper bound of z. 
# If z < lb[t,s], then v[t,s][z] = 0. 
# If z > ub[t,s], then v[t,s][z] = (z-ub[t,s] + v[t,s][ub[t,s]]).
# digit : Decimal digit of discretization.
function target_recursive(T,v,π_,t,s,z,mdp,lb,ub;digit=2)
    if haskey(v[t,s],z)
        return v[t,s][z]
    end
    # lower bound case
    if z < lb[t,s]
        return target_recursive(T,v,π_,t,s,lb[t,s],mdp,lb,ub;digit=digit)
    end
    # upper bound case
    if z > ub[t,s]
        return (target_recursive(T,v,π_,t,s,ub[t,s],mdp,lb,ub;digit=digit) + (z-ub[t,s]))
    end
    # final horizon case
    if t == (T+1)
        init_z = Base.max(z,0)
        for s in mdp.S
            v[t,s][z] = init_z
            π_[t,s][z] = 0 # no action (0) require at termination (T+1)
        end
        return init_z
    end
    q_min = Inf
    q_act = 0
    for a in mdp.valid_A[s]
        total = 0.0 
        for s_ in mdp.S_sa[s][a]
            total += (mdp.P[s, a, s_] * target_recursive(T,v,π_,t+1,s_, ceil((z-mdp.R[s, a, s_])/mdp.γ,digits=digit),mdp,lb,ub,digit=digit))
        end
        total *= mdp.γ

        if q_min > total
            q_min = total
            q_act = a
        end
    end

    v[t,s][z] = q_min
    π_[t,s][z] = q_act
    return q_min
end

# A function that collect solution of target value based MDP : 
# Average Value at Risk - Baurle (2011)
# Input : 
# mdp : A Markov Decision Process model
# T : Total time horizon
# obj : Currently only "CVaR" is handled
function targetVi(mdp::MDP,obj::Objective)
    if obj.ρ != "CVaR"
        error("targetVI can only handle CVaR, other utility function is not implemented.")
    end
    # compute value function lower bound and upper bound
    lb_v = VI(mdp,Objective(ρ="min",T = obj.T))["v"]
    ub_v = VI(mdp,Objective(ρ="max",T = obj.T))["v"]
    # set_Z : Initial targt value discretization
    minv = minimum(lb_v[1,findall(mdp.s0 .!= 0)])
    maxv = maximum(ub_v[1,findall(mdp.s0 .!= 0)])
    # digit(obj.δ) Integer : Decimal digit of discretization.
    digit = -ceil(Int,log(maxv-minv)/log(10))+obj.δ
    Z0 = ceil.(collect(minv:(10.0^(-digit)):maxv),digits=digit)
    # lb,ub : Lower bound and upper bound of z. During recursive
    # If z < lb[t,s], then v[t,s][z] = 0. 
    # If z > ub[t,s], then v[t,s][z] = (z-ub[t,s] + v[t,s][ub[t,s]]).
    lb = ceil.(lb_v,digits=digit)
    ub = ceil.(ub_v,digits=digit)
    # v : Value function 2D dictionaries v[t,s][z][1] (value), v[t,s][z][2] (action)
    v = [Dict{Float64, Float64}() for t in 1:(obj.T+1), s in 1:mdp.lSl]
    π_ = [Dict{Float64,Int}() for t in 1:(obj.T+1), s in 1:mdp.lSl]
    for z in ProgressBar(Z0)
        for s0 in findall(mdp.s0 .!= 0)
            target_recursive(obj.T,v,π_,1,s0,z,mdp,lb,ub,digit=digit)
        end
    end
    return Dict("v" => v, "π" => π_, "Z0" => Z0, "digit" => digit)
end

function initTarget(mdp::MDP, v::Matrix{Dict{Float64, Float64}},Z0::Vector{Float64}, parEval::Vector{Float64})
    S0 = findall(mdp.s0 .!= 0)
    v0 = [sum([v[1,s0][z] for s0 in S0] .* mdp.s0[S0]) for z in Z0] # v0[z] = E[ E[v[1,s0][z] | s̃0] ]
    opt_j = [argmax(Z0 .- (v0/(q))) for q in parEval]
    opt_value = Z0[opt_j] .- (v0[opt_j] ./ parEval) 
    return Dict("value"=> opt_value,"opt_z"=> Z0[opt_j], "α" => parEval ) # CVaR_value, z⋆ , α
end

function df2MDP(df,γ=0.95;s_init = 0)
    # read in unique states and actions then make them one-indexed
    S = unique([df.idstatefrom;df.idstateto])
    one_S = -minimum(S) + 1
    S .+= one_S
    df.idstatefrom .+= one_S
    df.idstateto .+= one_S

    A = unique(df.idaction)
    one_A = -minimum(A) + 1
    A .+= one_A
    df.idaction .+= one_A

    lSl = length(S)
    lAl = length(A)
    # Calculate transition and reward 
    P = zeros((lSl,lAl,lSl))
    R = zeros((lSl,lAl,lSl))
    if s_init == 0
        # s0 = ones(lSl) / lSl  # uniform initial distribution
        s0 = ones(lSl) ./ (lSl-1) # uniform initial distribution
        s0[end] = 0 # Sink state
    else
        ((s_init <= lSl) && (0 < s_init)) ||  error("Initial state error")
        s0 = zeros(lSl)
        s0[s_init] = 1.0
    end
    for i in eachrow(df)
        P[i.idstatefrom,i.idaction,i.idstateto] += i.probability
        R[i.idstatefrom,i.idaction,i.idstateto] += i.reward
    end
    # Divide to average model if given multi-models
    if hasproperty(df, :idoutcome)
        M = unique(df.idoutcome)
        lMl = len(M)
        P /= lMl
        R /= lMl
    end
    # Handle invalid actions (force a large negative rewards)
    valid_A = [[] for s in S]
    for s in S
        for a in A
            if sum(P[s,a,:]) == 0
                R[s,a,:] .= -Inf
                P[s,a,s] = 1.0
            else
                push!( (valid_A[s]), a)
            end
        end
    end

    (maximum(abs.(sum(P,dims=3) .- 1)) < 1e-14) || error("Transition does not sum to 1")
    S_sa = [ [ [sn for sn in S if P[s,a,sn] > 0] for a in A ] for s in S ]
    P_sa = [ [ [P[s,a,sn] for sn in S_sa[s][a]] for a in A ] for s in S ]
    for ps in P_sa
        for psa in ps
            (abs(sum(psa) - 1) < 1e-14) || error("Transition does not sum to 1")
        end
    end
    P_sample = [[Categorical(P[s,a,:]) for a in A] for s in S]
    return MDP(S, lSl, A, lAl, R, P, γ, s0,S_sa,P_sa,P_sample, valid_A)
end

# Define a helper function to sample from the Categorical distribution
function sample1_from_transition(P_sample,s, a)
    return rand(P_sample[s][a])
end

# Define a helper function to sample from the Categorical distribution
function sample_from_transition(P_sample,s, a, N)
    return rand(P_sample[s][a],N)
end




