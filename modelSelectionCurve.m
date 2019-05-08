function [polynomial_degree_vec, error_train, error_cv, error_test] = ...
				modelSelectionCurve(X, y, Xcv, ycv, Xtest, ytest, p, lambda_vec)

lambda_vec = lambda_vec';

polynomial_degree_vec = zeros(p, 1);
for i = 1:p
	polynomial_degree_vec(i) = i;
end;

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), p);
error_cv    = zeros(length(lambda_vec), p);
error_test  = zeros(length(lambda_vec), p);

options = optimset('MaxIter', 200, 'GradObj', 'on');

for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	for j = 1:p

		% Map X onto Polynomial Features and Normalize
		X_poly = polynomialFeaturesMap(X, j);
		mu = mean(X_poly);
		sigma = std(X_poly);
		X_poly = bsxfun(@rdivide, bsxfun(@minus, X_poly, mu), sigma);
		X_poly = [ones(size(X_poly, 1), 1), X_poly];                   % Add Ones

		% Map X_poly_cv and normalize (using mu and sigma)
		X_poly_cv = polynomialFeaturesMap(Xcv, j);
		X_poly_cv = bsxfun(@rdivide, bsxfun(@minus, X_poly_cv, mu), sigma);
		X_poly_cv = [ones(size(X_poly_cv, 1), 1), X_poly_cv]; % Add Ones

		% Map X_poly_test and normalize (using mu and sigma)
		X_poly_test = polynomialFeaturesMap(Xtest, j);
		X_poly_test = bsxfun(@rdivide, bsxfun(@minus, X_poly_test, mu), sigma);
		X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test]; % Add Ones
        
		% training model
		initial_theta     = zeros(size(X_poly, 2), 1); 
		costFunction      = @(t) linearRegressionCostFunction(X_poly, y, t, lambda);
		theta             = fmincg(costFunction, initial_theta, options);
        
		% compute error
		error_train(i, j) = linearRegressionCostFunction(X_poly, y, theta, lambda);
		error_cv(i, j)    = linearRegressionCostFunction(X_poly_cv, ycv, theta, lambda);
		error_test(i, j)  = linearRegressionCostFunction(X_poly_test, ytest, theta, lambda);
	end;
end;

error_train = error_train(:);
error_cv    = error_cv(:);
error_test  = error_test(:);

end