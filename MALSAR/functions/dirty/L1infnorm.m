function [Xnorm] = L1infnorm(X)
% ||X||_{1,2} = sum_i||X^i||_inf
Xnorm = sum(max(abs(X),[],2));
