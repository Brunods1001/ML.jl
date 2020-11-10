
function accuracy(y, ŷ)
    sum(y .== ŷ) / length(y)
end
