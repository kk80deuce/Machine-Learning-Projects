function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = X * theta;
error = h - y;
error_square = error.^2;
J = (1/(2*m))*sum(error_square);

grad_temp = (1/m) * (X' * error);
theta(1) = 0;
reg = (lambda/(2*m))*sum(theta.^2);
J = J + reg;

reg_grad = (lambda/m).*theta;
grad = grad_temp.+reg_grad;


 






% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
