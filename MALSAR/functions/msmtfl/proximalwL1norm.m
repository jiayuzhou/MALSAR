function [X] = proximalwL1norm(D, tau)
% min_X 0.5*||X - D||_F^2 + \sum_i tau_ij*|X_ij|
X = sign(D).*max(0,abs(D)- tau);
