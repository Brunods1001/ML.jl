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

mutable struct AdalineNN <: NeuralNetwork
    w::Array{Float64, 1}
    η::Float64
    n_iter::Int64
    function AdalineNN(η, n_iter)
        w = Array{Float64, 1}()
        new(w, η, n_iter)
    end
end

# Perceptron

function add_bias(X)
    bias = ones(size(X)[1])
    [bias X]
end

function update(p::Perceptron, X,  y)
    X_b = add_bias(X)
    ŷ = predict(p, X_b)
    Δw = p.η * transpose(y - ŷ) * X_b
    return transpose(Δw) + p.w
end

function update!(p::Perceptron, X, y)
    X_b = add_bias(X)
    ŷ = predict(p, X_b)
    Δw = p.η * transpose(y - ŷ) * X_b
    p.w =  transpose(Δw) + p.w
end

function update(p::AdalineNN, X, y)
    X_b = add_bias(X)
    ΔJ = transpose(transpose(y - X_b * p.w) * X_b)
    Δw = -p.η * ΔJ 
end

function update!(p::AdalineNN, X, y)
    X_b = add_bias(X)
    ΔJ = transpose(transpose(y - X_b * p.w) * X_b)
    Δw = -p.η * ΔJ
    p.w = p.w + Δw
end

function train!(p::NeuralNetwork, X, y)
    X_b = add_bias(X)
    if length(p.w) == 0
        p.w = Random.rand(size(X_b)[2])
    end
    for i in 1:p.n_iter
        update!(p, X, y)
    end
end

function predict(p::Perceptron, X)
    z = X * p.w
    ifelse.(z .≥ 0, 1, -1)
end

# AdalineNN
# Adaline uses a linear activation function to learn weights.
function linear_activation(p::AdalineNN)

end

