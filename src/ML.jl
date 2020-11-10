module ML

export Perceptron, 
       train!, 
       predict,
       split_data,
       strat_split,
       accuracy

include("perceptron.jl")
include("metrics.jl")
include("split.jl")

end # module
