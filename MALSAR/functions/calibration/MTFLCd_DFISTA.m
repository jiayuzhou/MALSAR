function [ W, info, Th ] = MTFLCd_DFISTA( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration
% diagnoal version for faster computation on small sample size.
%
% OBJECTIVE
%    min_W { sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
%
%  We solve this by the dual form
%    max_Theta min_W { sum_i^m theta_i^T (Xi wi - yi) + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
%          s.t.,     ||theta_i|| <= 1 (i = 1..m)
%
%
% INPUT
%  X - cell array of {n_i by d matrices} by m
%  y - cell array of {n_i by 1 vectors}  by m
%  lambda1 - regularization parameter of the l2,1 norm penalty
%  lambda2 - regularization parameter of the Fro norm penalty
%
% OUTPUT
%  W - task weights: d by t.
%  funcVal - the funcion value.
%
% Author: Jiayu Zhou, Pinghua Gong

%% Initialization
if(nargin<5), opts = []; end

opts = setOptsDefault( opts, 'verbose', 1); 
opts = setOptsDefault( opts, 'maxIter', 10000);
opts = setOptsDefault( opts, 'tol',     1e-7);
opts = setOptsDefault( opts, 'stopflag', 1);
verbose = opts.verbose;

info.algName = 'Dual FISTA';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', info.algName, opts.maxIter, opts.tol);
end

m = length(X); % task number
d = size(X{1}, 2);

% diagonalize X and vectorized y.
[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);

info.fvP = zeros(opts.maxIter, 1);
info.fvD = zeros(opts.maxIter, 1);
funcVal  = zeros(opts.maxIter, 1);
timeVal  = zeros(opts.maxIter, 1);

if isfield(opts, 'initTheta')
    Th0 = segL2Proj(opts.initTheta, Th_vecIdx);
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    Th0 = segL2Proj(randn(sum(samplesize), 1), Th_vecIdx);
end

%% Computation
if verbose == 1; fprintf('Iteration:     '); end

% bFlag = 0; % whether the gradient step only changes little.

Thk     = Th0;
Thk_old = Th0;

gamma = 1; gamma_inc = 2;

t = 1; t_old = 1;
for iter = 1: opts.maxIter
    iterTic = tic;
    alpha = (t_old  -1 )/t;
    Vk = Thk + alpha * (Thk - Thk_old);
    
    % function value and gradients of the search point. 
    [DVk, gDVk] = dualSmooth(Vk);
    
    for lsIter = 1: 100
        Thk  = segL2Proj(Vk + gDVk ./ gamma, Th_vecIdx);
        
        [DThk, gDThk, Wk, thNrms ] = dualSmooth(Thk);
        
        delta_ThkVk = Thk - Vk;
        
        %r_sum = norm(delta_ThkVk)^2;
        r_sum = sum(delta_ThkVk.^2);
        
%         % test if the gradient step makes little improvement
%         if (r_sum <=1e-20), bFlag=1; break; end
        
        if (DThk >= (DVk + gDVk' * delta_ThkVk - gamma/2 * r_sum)  )
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Thk_old = Thk;
    info.fvP(iter) = primalObjective(gDThk, thNrms);
    funcVal(iter)  = DThk;
    
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose >=2
        fprintf('%s: [Iteration %u][fvP %.4g][fvD %.4g]\n', ...
            info.algName, iter, info.fvP(iter), funcVal(iter));
    end
    
    % test stop condition.
 %   if (bFlag), break; end
    if iter>=2
        switch opts.stopflag
            case 1
                if (abs( funcVal(iter) - funcVal(iter-1) ) <= opts.tol* abs(funcVal(iter-1)))
                    break;
                end
            case 2
                if ~isfield(opts, 'obj')
                    error('opts.obj must be set');
                end
                if abs(info.fvP(iter) - opts.obj) <= opts.tol*opts.obj
                    break;
                end
            case 3
                if ~isfield(opts, 'W')
                    error('opts.W must be set');
                end
                if norm(Wk - opts.W,'fro') <= opts.tol*norm(opts.W,'fro')
                    break;
                end
             case 4
                 if abs(funcVal(iter) - info.fvP(iter)) <= opts.tol*abs(info.fvP(iter))
                     break;
                 end
        end
    end
    
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
end
if verbose == 1; fprintf('\n'); end

%% Output.
W   = Wk;
Th  = Thk;
info.funcVal = funcVal (1:iter); % dual
info.fvP     = info.fvP(1:iter);
info.fvD     = funcVal (1:iter);
info.timeVal = timeVal (1:iter);

%% Nested Functions
    function fvP = primalObjective(XWy, thNrms)
        % primal objective (NOTE: XWy is gradient gDThk)
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fvP = thNrms + segL2 (XWy, Th_vecIdx); 
    end

    function WTh = computeW(Th_vec)
        WTh = reshape(Th_vec' * Xdiag, d, m);% Compute UTh
        WTh = -1/lambda2*max(0,1-lambda1./repmat(sqrt(sum(WTh.^2,2)),1,m)).*WTh;
    end

    function [f, g, WTh, thNrms] = dualSmooth(Th_vec)
        WTh = computeW(Th_vec); % compute the corresponding W(Theta).
        % gradient. 
        g = Xdiag * WTh(:) - yvect; 
        % function value.
        thNrms = lambda1 * sum(sqrt(sum(WTh.^2, 2))) + lambda2 / 2 * sum(sum(WTh.^2));
        f = thNrms + Th_vec' * g;
    end

end
