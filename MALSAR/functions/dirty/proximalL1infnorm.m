function [X] = proximalL1infnorm(D, tau)
% min_X 0.5*||X - D||_F^2 + tau*||X||_{1,inf}
% where ||X||_{1,inf} = sum_i||X^i||_inf, where X^i denotes the i-th row of X

% X = D; n = size(D,2);
% for ii = 1:size(D,1)
%     [mu,~,~] = prf_lb(D(ii,:)', n, tau);
%     X(ii,:) = D(ii,:) - mu';
% end

[m,n]=size(D);
[mu,~,~]=prf_lbm(D,m,n,tau);
X = D - mu;