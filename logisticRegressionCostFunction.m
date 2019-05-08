function [J, grad] = logisticRegressionCostFunction(X, y, theta, lambda)
m = length(y); J = 0; grad = zeros(size(theta));
h = sigmoid(X * theta); theta(1) = 0;
J = (y' * log(h) + (1 - y)' * log(1 - h)) / (-m);
J += theta' * theta * lambda / 2 / m;
grad = ((X' * (h - y)) + lambda * theta) / m;
grad = grad(:);
end