function [error_train, error_cv, error_test] = learningCurve(J, X, y, Xcv, ycv, Xtest, ytest, lambda)
m = size(X, 1);
error_train = zeros(m, 1);
error_cv = zeros(m, 1);
error_test = zeros(m, 1);
initial_theta = zeros(size(X, 2), 1); 
options = optimset('MaxIter', 200, 'GradObj', 'on');
for i=1:m
	costFunction = @(t) J(X(1:i, :), y(1:i), t, lambda); % short hand
	theta = fmincg(costFunction, initial_theta, options); 
	error_train(i) = J(X(1:i,:), y(1:i), theta, lambda);
	error_cv(i) = J(Xcv, ycv, theta, lambda);
	error_test(i) = J(Xtest, ytest, theta, lambda);
end
end