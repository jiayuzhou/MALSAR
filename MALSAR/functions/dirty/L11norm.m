function [Xnorm] = L11norm(X)
% ||X||_tr = sum_i\sigma_i
Xnorm = sum(sum(abs(X)));
