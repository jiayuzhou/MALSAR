function [ fvP ] = primalObj( X, y, W, lambda1, lambda2)
%PRIMALOBJ Summary of this function goes here
%   Detailed explanation goes here

% primal objective (without augmented terms)
%  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2

fvP = lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2));
for i = 1: length(X)
    fvP = fvP + sqrt(sum((X{i} * W(:, i) - y{i}).^2));
end

end

