function [Xnorm] = wL1norm(X,lambda)
% ||X||_wL1 = sum_i \lambda_ij|X_ij|
Xnorm = sum(sum(lambda.*abs(X),2));
