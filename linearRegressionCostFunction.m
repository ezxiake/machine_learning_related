function [J, grad] = linearRegressionCostFunction(X, y, theta, lambda)
	m    = length(y);
	J    = (1/2/m) * sum( (X * theta - y).^2 ) + (lambda/2/m) .* sum( theta(2:end,:).^2 );
	% In the multivariate case, the cost function can also be written in the vectorized form.
	% J = 1/(2*m) * (X * theta - y)' * (X * theta - y); 
	grad = (1/m)   * sum( (X * theta - y).*X ) + (lambda/m)   .* [0, theta'(:,2:end)];
	grad = grad(:);
end