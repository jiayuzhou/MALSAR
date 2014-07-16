function [X] = proximalL11norm(D, tau)
% min_X 0.5*||X - D||_F^2 + tau*||X||_{1,1}
% where ||X||_{1,1} = sum_ij|X_ij|, where X_ij denotes the (i,j)-th entry of X
X = sign(D).*max(0,abs(D)-tau);
