function [W, info] = MTFLCd_ADMM_WSolver...
    (Xdiag, yvect, Thvect, Zvect, rho, lambda1, lambda2, opts)
%
% Multi-Task Feature Learning with Calibration - ADMM
% Subproblem: W
% diagnoalized version. 
% 
% Objective 
%   min_W {   rho/2 sum_i^m ||theta_i/rho - z_i + y+i - X_iwi|| 
%           + lambda2/2 ||W||_F^2 + lambda1 ||W||_{1,2}  
%          }
%
% INPUT
%  X - cell array of {n_i by d matrices} by m
%  y - cell array of {n_i by 1 vectors}  by m
%  Th - cell array of {n_i by 1 vectors} by m
%  Z  - cell array of {n_i by 1 vectors} by m
%  rho - parameter of the augmented Lagrange
%  lambda1 - regularization parameter of the l2,1 norm penalty
%  lambda2 - regularization parameter of the Fro norm penalty
%  opts - MANDATORY optimization options. 
%
% OUTPUT
%  W - task weight d by m.
%  
% Author: Jiayu

%% Initialization

W0 = opts.init; % the outer loop must pass in the last solution 

funcVal = zeros(opts.maxIter,1);

TrZy = Thvect/rho - Zvect + yvect;

%% Computation

bFlag = 0; 

W     = W0;
W_old = W0;

gamma = 1; gamma_inc = 2;

t =1; t_old = 1; 
for iter = 1: opts.maxIter
    alpha = (t_old  -1 )/t;
    V = W + alpha * (W - W_old);
    
    [fV, gV] = smoothObj(V);
    
    for lsIter = 1:100
        W = proj(V - gV/gamma, lambda1/gamma);
        f = smoothObj(W);
        
        delta_W = W - V;
        r_sum = sum(sum(delta_W.^2));
        
        if(r_sum <= 1e-20), bFlag = 1; break; end
        
        if f<= fV + sum(sum(delta_W .* gV)) + gamma/2 * r_sum
            break;
        end
        
        gamma = gamma * gamma_inc;
    end
    
    W_old = W;
    
    funcVal(iter) = f + lambda1 * sum(sqrt(sum(W.^2, 2)));
    
    if(bFlag), break; end
    
    % convergence
    if(iter>1)
        if (abs( funcVal(iter) - funcVal(iter-1) ) ...
                <= opts.tol* abs(funcVal(iter-1)))
            break;
        end
    end
    
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
end

info.funcVal = funcVal(1:iter);

%% Nested functions
    function [X] = proj(D, t) % l2.1 norm projection. 
        X = repmat(max(0, 1 - t./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
    end

    function [f, g] = smoothObj(W)
        f = 0.5 * lambda2 *sum(sum(W.^2));
        tpVect = TrZy - Xdiag * W(:);
        f = f +  0.5 * rho * sum(tpVect.^2);
        if nargout >= 2
            g = reshape(lambda2 * W(:) - rho * (Xdiag' * tpVect), size(W));
        end
    end

end