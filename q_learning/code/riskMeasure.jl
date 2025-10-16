using Statistics
using LogExpFunctions

"""
check_size(X,prob): takes in random variables X and optional probability prob
1. if no probability is given, uniform distribution is assumed.
2. if X and prob are of different sizes, error is thrown.
3. remove values with probability 0.
"""
function check_size(X::Array,prob::Array=[])
    if length(prob) == 0
        prob = ones(size(X)...) / length(X)
    else
        size(X) == size(prob)  || error("X and prob must have the same size")
        X = X[prob .> 0]
        prob = prob[prob .> 0]
    end
    return X, prob
end

# sort distribution based on X
function sortByValue(X,p)
    order = sortperm(X)
    return (X[order],p[order])
end

# Normalization of probabilities so they sum to 1
function normalizeProbability(p::Array)
    p ./= sum(p)
    probSum = sum(p)
    if probSum < 1
        p ./= probSum
    end
    return p
end

mutable struct distribution
    X::Array
    p::Array
    cdf::Array
    XᵀP::Array
    function distribution(X::Array,p::Array=[])
        X,p = check_size(X,p)
        X,p = sortByValue(X,p)
        p = normalizeProbability(p)
        new(X,p,cumsum(p),[])  # Calls the default constructor
    end
end


"""
E(d): takes in distribution w/ values (X) and probability (p)
and returns the expected value of the distribution.
"""
function E(d::distribution)
    return sum(d.X .* d.p)
end

"""
min(d): takes in distribution w/ values (X) and probability (p)
and returns the minimum of X that is possible to occur.
"""
function min(d::distribution)
    return minimum(d.X)
end

"""
max(d): takes in distribution w/ values (X) and probability (p)
and returns the maximum of X that is possible to occur.
"""
function max(d::distribution)
    return maximum(d.X)
end

"""
quant(d,α) -> torch.Tensor:  float: Search over cdf.
Takes in an array of risk level (α).
Return the (α)-th quantile of the distribution (d).
"""

function quant(d::distribution, α::Array)
    i = Base.min.(searchsortedfirst.(Ref(d.cdf), α), length(d.cdf))
    return d.X[i]
end

"""
VAR(d,α) -> torch.Tensor:  float: Search over cdf.
Takes in an array of risk level (α).
Return the (α)-th quantile of the distribution (d).
"""

function VaR(d::distribution, α::Array)
    i = Base.min.(searchsortedlast.(Ref(d.cdf), α) .+ 1, length(d.cdf))
    return d.X[i]
end

"""
CVaR(d,α) -> float: Search over cdf.
Takes in an array of risk level (α).
Return the (α)-CVaR of the distribution (d).
"""

function CVaR(d::distribution, α::Array)
    d.XᵀP = cumsum(d.X .* d.p)
    i = Base.min.(searchsortedfirst.(Ref(d.cdf), α), length(d.cdf))
    return ifelse.(i .> 1, (d.XᵀP[i] .+ d.X[i] .* (α .- d.cdf[i])) ./ α, d.X[1])
end

"""
erm(d,β)   =  log(mean(exp( -β*X )))/-β
            =  log(sum(exp(-β*X + log(p))))/-β
            =  LSE(-β*X + log(p))/-β
"""
function ERMs(d::distribution, β::Array)
    ermValue = logsumexp(( -β .* d.X' ) .+ log.( d.p' ), dims=2) ./ (-β)

    println("----test values :  ", ermValue)
    return ifelse.(β .!= 0, ermValue, E(d))
end

function base_ERM(X::Array,p::Array, β::Float64)
    if β == 0
        return sum(X .* p)
    else
        return logsumexp(( -β .* X ) .+ log.( p ), dims=1)[1] / (-β)
    end
end

function ERM(d::distribution, β::Float64)
    return base_ERM(d.X ,d.p ,β)
end






"""
evar_discretize(α, δ, ΔR)

Computes an optimal grid of β values in the definition of EVaR for
the risk level `α` (smaller is more risk averse) with the error guaranteed to
be less than `δ`. The range of returns (maximum - minumum possible) must
be bounded by `ΔR`
"""
function evar_discretize(α::Real, δ::Real, ΔR::Real)
    zero(δ) < δ  || error("δ must be > 0")
    # set the smallest and largest values
    β1 = 8*δ / (ΔR^2)
    βK = Base.max(-log(α) / δ,1.0)
    βs = Vector{Float64}([])
    β = β1

    while β < βK
        push!(βs, β)
        β *= (log(α) / (β*δ + log(α)))
    end
    return βs
end

"""
EVaR(d,a) = sup_b( erm(d,b) + log(a)/b )
ERM solution is between [minimum and mean]
"""
function EVaR(d::distribution, α::Array;δ::Number = 0.1,min_α = 0.01,fast_approx=true)
    min_val = min(d)
    mean_val = E(d)
    if fast_approx
        β = 1000 * (0.999 .^ (0:20000))
    else 
        max_val = max(d)
        α_min = Base.max(minimum(α[ α .> 0]),min_α)
        β = evar_discretize(α_min, δ,max_val-min_val)
    end
    erm_val = ERMs(d, β)
    return clamp.(maximum( (erm_val' .+ log.(α) ./ β' ) ,dims=2),min_val,mean_val) 
end


"""
jointD(d_conds) 
Takes in a list of values (X), conditional prob (con_pr) and marginal prob (p):
1. Combines the conditional probs and marginal to get joint probability.
2. Use the values (X) and joint probability to create the joint distribution dataframe (d).
"""

function jointD(Xs::Vector{Vector{Float64}},con_pr::Vector{Vector{Float64}},marg_pr::Vector{Float64})
    ((length(con_pr) == length(marg_pr)) && (length(con_pr) == length(Xs))) || error("distributions and marginal_pr must have the same size")
    for (i,p) in enumerate(marg_pr)
        con_pr[i] *= p # joint_pr = cond_pr * marg_pr
    end
    return distribution(reduce(vcat,Xs), reduce(vcat,con_pr))
end

"""
jointD_fixPDF(d_cond) 
Takes in a list of values (X), a fix conditional prob (p_cond) and marginal prob (p):
1. Combines the conditional probs and marginal to get joint probability.
2. Use the values (X) and joint probability to create the joint distribution dataframe (d).
"""

function jointD_fixPDF(Xs::Vector{Vector{Float64}},d_cond::Vector{Float64},marg_pr::Vector{Float64})
    (length(Xs) == length(marg_pr)) || error("distributions and marginal_pr must have the same size")
    con_pr = [d_cond * p for p in marg_pr]
    X = reduce(vcat,Xs)
    p = reduce(vcat,con_pr)
    (length(X) == length(p)) || error("random variable and probabilities must have the same size")
    return distribution(X,p)
end

"""
VaR_cdf2pdf(cdf) - convert cdf to pdf (underestimation)
"""
function q_evenPdf(cdf)
    return ones(size(cdf)) ./ length(cdf)
end

"""
VaR_cdf2pdf(cdf) - convert cdf to pdf (underestimation)
"""
function VaR_cdf2pdf(cdf)
    return diff([cdf; 1])
end
"""
VaR2D(X,cdf,pdf) - underestimation of the true distribution
Takes in an array of sorted X values its respective cdf/pdf:
1. Calcaulate the pmf from cdf :diff(cdf, append=1).
returns the distribution dataframe (d).
"""
function VaR2D(X::Array, cdf::Array)
    issorted(X) ||  error("values of var must be sorted")
    return distribution(X, VaR_cdf2pdf(cdf))
end


"""
quant_cdf2pdf(cdf) - convert cdf to pdf (overestimation)
"""
function quant_cdf2pdf(cdf)
    return diff([0;cdf])
end
"""
quantile2D(X,cdf,pdf) - overestimation of the true distribution
Takes in an array of sorted X values its respective cdf/pdf:
1. Calcaulate the pmf from cdf, diff(cdf, prepend=0).
returns the distribution (d).
"""
function quantile2D(X::Array, cdf::Array)
    issorted(X) ||  error("values of var must be sorted")
    return distribution(X, quant_cdf2pdf(cdf))
end

"""
CVaR_cdf2pdf(cdf) - convert cdf to pdf
"""
function CVaR_cdf2pdf(cdf)
    return diff(cdf)
end

"""
CVaR2X(cvar,cdf,pdf)
Takes in an array of sorted CVaR values its respective cdf:
1. Calcaulate the X from cvar and cdf, np.diff(cdf * cvar)/p.
returns the distribution (d).
"""
function CVaR2X(cvar::Array, cdf::Array, pdf::Array; decimal::Integer = 10)
    return round.( diff(cdf .* cvar) ./ pdf, digits=decimal)
end
"""
CVaR2D(cvar,cdf)
Takes in an array of sorted CVaR values its respective cdf:
1. Calcaulate the pmf from cdf, pdf = np.diff(cdf).
1. Calcaulate the X from cvar and cdf, np.diff(cdf * cvar)/p.
returns the distribution (d).
"""
function CVaR2D(cvar::Array, cdf::Array; decimal::Integer = 10)
    issorted(X) ||  error("values of cvar must be sorted")
    pdf = diff(cdf)
    X = CVaR2X(cvar,cdf,pdf,decimal=decimal)
    return distribution(X, pdf)
end

"""
extreme(vector)
take in a vector of values and return (min,max) of the vector
"""
function extreme(X)
    return (minimum(X),maximum(X))
end


