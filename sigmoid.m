function g = sigmoid(z)
    % Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    g = 1.0 ./ (1.0 + exp(-z));
end