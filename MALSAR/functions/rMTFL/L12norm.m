function [Xnorm] = L12norm(X)
% ||X||_{1,2} = sum_i||X^i||_2
Xnorm = sum(sqrt(sum(X.^2,2)));
