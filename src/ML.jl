module ML

export Perceptron, 
       AdalineNN,
       train!, 
       predict,
       split_data,
       strat_split,
       accuracy,
       SSE,
       update

include("cost.jl")
include("perceptron.jl")
include("metrics.jl")
include("split.jl")

end # module
