
"""
Train test split
"""
function split_data(X, y, splits = [0.7, 0.15, 0.15];
               shuffle = true,
               seed = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    if shuffle
        idx = Random.shuffle(1:size(X)[1])
        X = X[idx, :]
        y = y[idx, :]
    end
    m = size(X)[1]
    X_splits = []
    y_splits = []
    m_start = 1
    for split in splits[1:end - 1]
        m_end = m_start + Int64(round(split * m)) - 1
        X_split = X[m_start:m_end, :]
        y_split = y[m_start:m_end]
        push!(X_splits, X_split)
        push!(y_splits, y_split)
        m_start = m_end + 1
    end
    X_split = X[m_start:end, :]
    y_split = y[m_start:end]
    push!(X_splits, X_split)
    push!(y_splits, y_split)
    return X_splits, y_splits
end


"""
Stratify split
"""
function strat_split(X, y, splits = [0.7, 0.15, 0.15];
                     shuffle = true,
                     seed = nothing)
    m, n = size(X)
    y_cat = unique(y)
    X_splits = repeat([reshape([], 0, n)], length(splits))
    y_splits = repeat([[]], length(splits))
    for yk in y_cat
        idx = y .== yk
        X_splits_k, y_splits_k = split_data(X[idx, :], y[idx], splits;
              shuffle = shuffle, seed = seed)
        for i in 1:length(splits)
            X_splits[i] = vcat(X_splits[i], X_splits_k[i])
            y_splits[i] = vcat(y_splits[i], y_splits_k[i])
        end
    end
    return X_splits, y_splits
end

