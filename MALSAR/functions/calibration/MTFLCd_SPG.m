function [ W, info ] = MTFLCd_SPG( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration - Smoothing Proximal
% Gradient (SPG)
%
% diagnoalized
%
% OBJECTIVE
%    min_W { sum_i^m ||Xi wi - yi||_mu + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
%        ||Xi wi - yi||_mu = max <vi, Xi wi - yi> s.t. ||vi||<=1.
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

% 0 nothing. 1 iteration num. 2 iteration details.
opts = setOptsDefault( opts, 'verbose', 1); 
opts = setOptsDefault( opts, 'maxIter', 10000);
opts = setOptsDefault( opts, 'tol',     1e-7);
opts = setOptsDefault( opts, 'epsilon', 1e-1);
opts = setOptsDefault( opts, 'stopflag', 1);
verbose = opts.verbose;

info.algName = 'Smth SpaRSA';

if verbose > 0
    fprintf('%s Config [MaxIter %u][Tol %.4g][Epsilon %.4g]\n', ...
        info.algName, opts.maxIter, opts.tol, opts.epsilon);
end

m = length(X); % task number
d = size(X{1}, 2);

% diagonalize X and vectorized y.
[Xdiag, ~, Th_vecIdx, yvect] = diagonalize(X, y);

sigma = 1e-3;    % line search constant.
eta_init = 1;    % initial value of eta (1 = beta^0).
eta_max  = 1e+14;
eta_min  = 1e-14;
beta = 1.15;     % eta incremental
mu =  opts.epsilon/m;  
muInc = 1;     % decrease of mu.

if isfield(opts, 'initW')
    W0 = opts.initW;
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    W0 = randn(d, m); % starting from a random point.
end

info.fvP = zeros(opts.maxIter, 1);
funcVal  = zeros(opts.maxIter, 1);
timeVal  = zeros(opts.maxIter, 1);

lsCnt = 0; % count for line search. 

%% Computation
if verbose == 1; fprintf('Iteration:     '); end

Wk = W0;

for iter = 1: opts.maxIter
    iterTic = tic;
    
    [ fWk_mu , gWk_mu] = smoothObjective(Wk, mu);
    
    eta = eta_init;
    if iter >1  % BB-rule.
        xx = Wk - Wk_old; yy = gWk_mu - gWk_mu_old;
        eta = sum(sum(xx .* yy))/sum(sum(xx .* xx));
        eta = min(eta_max, max(eta_min, eta));
    end
    for lsIter = 1:100
        % smooth projection
        Wk_new = proj(Wk - gWk_mu / eta, lambda1/eta);
        [fWk_new_mu, XWy, thNrms ] = smoothObjectiveFv(Wk_new, mu);
        
        % line search
        if (fWk_mu - fWk_new_mu) >= sigma * eta /2 * sum(sum((Wk_new - Wk).^2))
            break;
        end
        
        eta = eta * beta;
    end
    lsCnt = lsCnt + lsIter;
    
    Wk_old     = Wk;
    gWk_mu_old = gWk_mu;
    Wk = Wk_new;
    
    funcVal(iter)  = fWk_new_mu; % the smoothed objective function
    info.fvP(iter) = primalObjective(XWy, thNrms); % the primal function.
    
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose ==2
        fprintf('%s: [Iteration %u][fvP %.4g][fvS %.4g][LS %u]\n', ...
            info.algName, iter, info.fvP(iter), funcVal(iter), lsCnt);
    end
    
    % test stop condition.
    if iter>=2
        switch opts.stopflag
            case 1
                if (abs( funcVal(iter) - funcVal(iter-1) ) <=...
                        opts.tol* abs(funcVal(iter-1)))
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
        end
    end
    
    mu = mu * muInc;
end
if verbose == 1; fprintf('\n'); end

%% Output.
W   = Wk;
info.funcVal = funcVal (1:iter);
info.fvP     = info.fvP(1:iter);
info.timeVal = timeVal (1:iter);


%% Nested Functions
    function fv = primalObjective(XWy, thNrms)
        % primal objective
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fv =  segL2 (XWy, Th_vecIdx) + thNrms;
    end

    function [f, g] = smoothObjective(W, mu)
        % f: funcVal of the ENTIRE smoothing function (including l21).
        % g: gradient of the smooth part of the smoothing function.
        
        XWy = Xdiag * W(:) - yvect;
        vv = segL2Proj (XWy/mu, Th_vecIdx);
        f = lambda2 / 2 * sum(sum(W.^2))  + lambda1 * L21norm(W) ...
            + vv' * XWy - mu/2 * sum(vv.^2);
        
        g = lambda2 * W + reshape( Xdiag' * vv, size(W)); 
    end

    function [f, XWy, thNrms] = smoothObjectiveFv(W, mu)
        % f: funcVal of the ENTIRE smoothing function (including l21).
        thNrms = lambda2 / 2 * sum(sum(W.^2))  + lambda1 * L21norm(W);
        
        % NOTE:there is a 1e-15 difference from matlab results on gradient.
        XWy = Xdiag * W(:) - yvect;
        vv = segL2Proj (XWy/mu, Th_vecIdx);
        f = thNrms + vv' * XWy - mu/2 * sum(vv.^2);
    end

end

function [Xnrm] = L21norm(X)
% ||X||_{1,2} = sum_i ||X^i||_2
Xnrm = sum(sqrt(sum(X.^2, 2)));
end

function [X] = proj(D, lambda )
% l2.1 norm projection.
X = repmat(max(0, 1 - lambda./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
end
