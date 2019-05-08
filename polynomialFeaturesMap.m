function X_poly = polynomialFeaturesMap(X, p)
n = size(X, 2);
% X1, X1.^2, X1.^3, etc...
if n == 1,
	X_poly = zeros(numel(X), p);
	for j = 1:p
		X_poly(:,j) = X .^ j;
	end;
	%X_poly = [ones(size(X_poly),1) X_poly];
	%disp(n);
% X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
elseif n == 2,
	degree = p;
	X1 = X(:,1);
	X2 = X(:,2);
	X_poly = ones(size(X1(:,1)));
	for i = 1:degree
		for j = 0:i
			X_poly(:, end+1) = (X1.^(i-j)).*(X2.^j);
		end
	end;
	%disp(n);
else 
	X_poly = 0;
	%disp(n);
end;
end