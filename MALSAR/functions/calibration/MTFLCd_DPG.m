function [ W, info, Th ] = MTFLCd_DPG( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration
% diagnoal version for faster computation on small sample size.
% dual projected gradient. 
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

info.algName = 'Dual SpaRSA';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', info.algName, opts.maxIter, opts.tol);
end

m = length(X); % task number
d = size(X{1}, 2);

% diagonalize X and vectorized y.
[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);

sigma = 1e-3;    % line search constant.
eta_init = 1;    % initial value of eta (1 = beta^0).
eta_max  = 1e+14;
eta_min  = 1e-14;
beta = 2;     % eta incremental

nonMonotoneNum = 0;

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

Thk  = Th0;

for iter = 1: opts.maxIter
    iterTic = tic;
    
    [DThk, gDThk] = dualSmooth(Thk);
    
    eta = eta_init;
    if iter >1  % BB-rule.
        xx = Thk - Thk_old; yy = gDThk - gDThk_old;
        eta = -sum(sum(xx .* yy))/sum(sum(xx .* xx)); % 
        eta = min(eta_max, max(eta_min, eta));
    end
    for lsIter = 1: 100
        Thk_new  = segL2Proj(Thk + gDThk / eta, Th_vecIdx);
        [DThk_new, gDThk_new, Wk, thNrms ] = dualSmooth(Thk_new);
        
        % line search 
        dd = min([funcVal(max(1, iter - nonMonotoneNum + 1): iter); DThk]);
        if (dd - DThk_new) <= -sigma * eta /2 * sum(sum((Thk_new - Thk).^2))
            break;
        end
            
        eta = eta * beta;
    end
    
    Thk_old   = Thk;
    gDThk_old = gDThk;
    Thk = Thk_new;
    
    fvP = primalObjective(gDThk_new, thNrms);
    info.fvP(iter) = fvP;
    % modified by Pinghua
    funcVal(iter)  = DThk_new;
    % funcVal(iter)  = DThk;
    
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose >=2
        fprintf('%s: [Iteration %u][fvP %.4g][fvD %.4g]\n', info.algName, iter, fvP, DThk);
    end
    
    % test stop condition.
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
end
if verbose == 1; fprintf('\n'); end

%% Output.
W   = Wk;
Th  = Thk_new;
info.funcVal = funcVal (1:iter);
info.fvP     = info.fvP(1:iter);
info.fvD     = funcVal (1:iter);
info.timeVal = timeVal (1:iter);

%% Nested Functions
    function fvP = primalObjective(XWy, thNrms)
        % primal objective
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fvP = thNrms + segL2 (XWy, Th_vecIdx);
    end

    function WTh = computeW(Th_vec)
        WTh = reshape(Th_vec' * Xdiag, d, m);% Compute UTh
        WTh = -1/lambda2*max(0,1-lambda1./repmat(sqrt(sum(WTh.^2,2)),1,m)).*WTh;
    end

    function [f, g, WTh, thNrms] = dualSmooth(Th_vec)
        WTh = computeW(Th_vec); % compute the corresponding W(Theta).
        % gradient 
        g = Xdiag * WTh(:) - yvect;
        % function value.
        thNrms = lambda1 * sum(sqrt(sum(WTh.^2, 2))) + lambda2 / 2 * sum(sum(WTh.^2));
        f = thNrms + Th_vec' * g;
    end

end

