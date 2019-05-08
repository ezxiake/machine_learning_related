function [theta, J_history] = userDefinedGradientDescent(f, theta, alpha, num_iters)
    J_history = zeros(num_iters, 1);
    for iter = 1:(num_iters)
        % Save the cost J in every iteration    
        [J_history(iter), grad] = f(theta);
        theta = theta - alpha .* grad;
    end
end