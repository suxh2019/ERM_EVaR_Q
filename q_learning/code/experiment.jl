include("TabMDP.jl")
include("utils.jl")
include("onlineMDP.jl")

function delete_risk_VI(ρ,vf;mdp_dir = "experiment/domain/MDP/")
    domains = readdir(mdp_dir)
    for d in domains
        domain = d[1:end-5]
        delete!(vf["1"][domain], ρ)
    end 
    return vf
end

function solveVI(objs::Vector{Objective};mdp_dir = "experiment/domain/MDP/",filename = "experiment/run/train/out.jld2",cache=true)
    # load domains file name, and load value function dictionaries
    domains = readdir(mdp_dir)
    vf = init_jld(filename)
    for d in domains
        # load mdp given domain nam
        domain = d[1:end-5]
        mdp = load_jld(mdp_dir * d)["MDP"]
        # for each algorithm of interest, train via value iteration and save 
        add = false # boolean variable check whether we have added any new element to the dictionary
        for obj in objs
            if (!cache) || (!in_jld(vf,obj.l,domain,obj.ρ)) # If VI output not exist, train and save
                println("Working on solveVI : ",domain ,"($(mdp.lSl),$(mdp.lAl)) for ",obj.ρ)
                insert_jld(vf , obj.l, domain, obj.ρ,  VI(mdp,obj)) # train and save VI-output
                add = true
            else
                println(obj.l,"quantiles    for ",domain,"  existed for ",obj.ρ)
            end 
        end
        if add # if we have added new element, we would rewrite the updated dictionary to its path
            save_jld(filename,vf)
        end
    end
    return vf
end

function evaluations(vf,objs::Vector{Objective};ENV_NUM = 10000, T_inf = 500,
    mdp_dir = "experiment/domain/MDP/",testfile = "experiment/run/test/evals.jld2",
    seed=0,quant_ϵ=1e-14)
    domains = readdir(mdp_dir)
    evals = init_jld(testfile)
    for d in domains 
        domain = d[1:end-5]
        mdp = load_jld(mdp_dir * d)["MDP"]
        for obj in objs
            println("Working on evaluations : ",domain ,"($(mdp.lSl),$(mdp.lAl)) for $(obj.ρ)")
            in_jld(vf, obj.l, domain, obj.ρ) || error("Could not find setting ($(obj.l),$domain,$(obj.ρ)) in vf")
            insert_jld(evals , obj.l, domain, obj.ρ,  
            simulate(vf[string(obj.l)][domain][obj.ρ],obj,mdp,ENV_NUM=ENV_NUM,T = (obj.T == -1 ? T_inf : obj.T) ,seed=seed,quant_ϵ=quant_ϵ))
        end
    end
    save_jld(testfile,evals)
end

function simplifyEvals(objs::Vector{Objective};mdp_dir = "experiment/domain/MDP/",testfile = "experiment/run/test/evals.jld2",eval_metric = VaR)
    domains = readdir(mdp_dir)
    evals = init_jld(testfile)
    VaR_results = Dict()
    for d in domains 
        domain = d[1:end-5]
        VaR_results[domain] = Dict()
        for obj in objs
            in_jld(evals, obj.l, domain, obj.ρ) || error("Could not find setting in evals $domain $(obj.l) $(obj.ρ)")
            println("Working on simplifyEvals : ",domain ," for $(obj.ρ)")
            if obj.ρ_type == "quant" 
                for opt in ["option 1"] 
                    x = evals[string(obj.l)][domain][obj.ρ][opt]
                    VaR_results[domain][obj.ρ] = Dict("values"=>[eval_metric(distribution(v),[lvl])[1] for (v,lvl) in zip(x["values"],x["α"])],"α"=>x["α"])
                end
            else
                x = evals[string(obj.l)][domain][obj.ρ]
                if obj.ρ in Set(["E","mean","min","max"]) # for mean there is only a single policy regarding parameter, can apply one evaluation for all levels
                    VaR_results[domain][obj.ρ] = Dict("values"=>eval_metric( distribution(x["values"]), obj.parEval ),"α"=>obj.parEval)
                else
                    VaR_results[domain][obj.ρ] = Dict("values"=>[eval_metric(distribution(v),[lvl])[1] for (v,lvl) in zip(x["values"],x["α"])],"α"=>x["α"])
                end
            end
        end
    end
    return VaR_results
end


function getTargetVaR(vf,objs::Vector{Objective};mdp_dir = "experiment/domain/MDP/")
    domains = readdir(mdp_dir)
    targetVaR = Dict()
    for d in domains 
        domain = d[1:end-5]
        mdp = load_jld(mdp_dir * d)["MDP"]
        targetVaR[domain] = Dict()
        for obj in objs
            v = vf[string(obj.l)][domain][obj.ρ]["v"]
            if obj.T == -1
                d0 = initDistribution(mdp,v, obj.pdf)
            else
                d0 = initDistribution(mdp,v[1,:,:], obj.pdf)
            end
            targetVaR[domain][obj.ρ] = Dict("values"=>d0.X,"α"=>d0.cdf)
        end
    end
    return targetVaR
end

