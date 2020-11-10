using RDatasets


iris = dataset("datasets", "iris")

X = select(iris, Not("Species"))
X = convert(Array{Float64, 2}, X)
y = iris["Species"]
y = ifelse.(y .== "setosa", 1, -1)
