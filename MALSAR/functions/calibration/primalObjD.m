function [ fvP ] = primalObjD( Xdiag, yvect, Th_vecIdx, lambda1, lambda2, W )
% primal objective (diagonalized)
%  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
fvP = lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2))...
    + segL2 (Xdiag * W(:) - yvect, Th_vecIdx);

end

