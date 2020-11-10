import Random

abstract type Model end

abstract type NeuralNetwork end


mutable struct Perceptron <: NeuralNetwork
    w::Array{Float64, 1}
    η::Float64
    n_iter::Int64
    function Perceptron(η, n_iter)
        w = Array{Float64, 1}()
        new(w, η, n_iter)
    end
end

function add_bias(X)
    bias = ones(size(X)[1])
    [bias X]
end


function update(p::Perceptron, X,  y)
    ŷ = predict(p, X)
    Δw = p.η * transpose(y - ŷ) * X
    return transpose(Δw) + p.w
end

function update!(p::Perceptron, X, y)
    ŷ = predict(p, X)
    Δw = p.η * transpose(y - ŷ) * X
    p.w =  transpose(Δw) + p.w
end

function train!(p::Perceptron, X, y)
    X_b = add_bias(X)
    if length(p.w) == 0
        p.w = Random.rand(size(X)[2])
    end
    for i in 1:p.n_iter
        update!(p, X, y)
    end
end

function predict(p::Perceptron, X)
    z = X * p.w
    ifelse.(z .≥ 0, 1, -1)
end


