function [X] = proximalL12norm(D, tau)
% min_X 0.5*||X - D||_F^2 + tau*||X||_{1,2}
% where ||X||_{1,2} = sum_i||X^i||_2, where X^i denotes the i-th row of X
X = repmat(max(0, 1 - tau./sqrt(sum(D.^2,2))),1,size(D,2)).*D;