using JLD2
using JuliennedArrays

################################################################
# --------------------- Eps Decay Fun --------------------------
# expDecay: exponential decay function
################################################################

function noDecay(eps_min, eps_cur, nlearn, decay_rate)
    return eps_cur
end

function linearDecay(eps_min, eps_cur, nlearn, decay_rate)
    return Base.max(eps_min, eps_cur - decay_rate)
end

function harmonicDecay(eps_min, eps_cur, nlearn, decay_rate)
    decay_factor = 1.0 / decay_rate
    return Base.max(eps_min, eps_cur * (nlearn - 1 + decay_factor ) / (nlearn  + decay_factor))
end

function expDecay(eps_min, eps_cur, nlearn, decay_rate)
    return Base.max(eps_min, eps_cur * (0.1 ^ (decay_rate)))
end

################################################################
# --------------------- Counter class --------------------------
# keep track of update num and harmonic decaying epsilon function
# eps_min: stop decaying when epsilon is leq to eps_min
# decay_rate: numerator of harmonic decay
# nlearn: number of learning performed
################################################################
mutable struct Counter
    ϵ_min::Float64
    decay_fun::Function
    decay_rate::Float64
    ϵ::Float64
    nlearn::Int
    stop::Bool
    function Counter(ϵ_min, decay_fun; decay_rate=1e-4, ϵ=1.0, nlearn=0)
        ϵ = Base.max(ϵ_min,ϵ)
        new(ϵ_min,decay_fun,decay_rate,ϵ,nlearn,false)
    end
end

function increase(cnt::Counter)
    cnt.nlearn += 1
    if !cnt.stop
        cnt.ϵ = cnt.decay_fun(cnt.ϵ_min, cnt.ϵ, cnt.nlearn, cnt.decay_rate)
        cnt.stop = (cnt.ϵ <= cnt.ϵ_min)
    end
end

function reset(cnt::Counter)
    cnt.nlearn = 0
    cnt.stop = false
end

################################################################
# check_path(directory):
# Take in a path check directory existence
# if not exist, create that directory. Return the path.
################################################################

function check_path(path::AbstractString)
    directory = join(split(path, "/")[begin:end-1],"/")
    if (directory != "") && (!isdir(directory))  # Check if the directory does not exist
        mkpath(directory) 
    end
    return path
end

################################################################
# save_jld(object,filename):
# Store the object with pickle.
#     does not return
################################################################

function save_jld(filename,object)
    check_path(filename)
    save(filename,object)
end


################################################################
# load_jld(filename):
# retrieve the object with pickle.
#     return the object.
################################################################

function load_jld(filename)
    isfile(filename) || error("The file ",filename," does not exist.")
    return load(filename)
end

################################################################
# apply(function, X::Array; dims=0)
# Apply a function over dims of X  and the drop the dims
################################################################
function apply(fx::Function,X;dims::Int=0)
    # return dropdims(mapslices(fx,X,dims=dims),dims=dims)
    return map(fx,JuliennedArrays.Slices(X,dims))
end

function apply!(fx::Function,destination,X;dims::Int=0)
    map!(fx,destination,JuliennedArrays.Slices(X,dims))
    return destination
end

################################################################
# dummy function is a function that takes in anything and returns first arg
################################################################
function dummy(x=nothing,y=nothing) return x end

################################################################
# clamp a value x to be between m and M
################################################################
clamp(x, m, M) = Base.max(m, Base.min(x, M)) # m ≤ x ≤ M 



