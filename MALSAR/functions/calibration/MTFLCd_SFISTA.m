function [ W, info ] = MTFLCd_SFISTA( X, y, lambda1, lambda2, opts )
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

info.algName = 'Smth FISTA';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g][Epsilon %.4g]\n', ...
        info.algName, opts.maxIter, opts.tol, opts.epsilon);
end

m = length(X); % task number
d = size(X{1}, 2);

% diagonalize X and vectorized y.
[Xdiag, ~, Th_vecIdx, yvect] = diagonalize(X, y);

mu = opts.epsilon/m;  
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

bFlag = 0; 

W     = W0;
W_old = W0;

t =1; t_old = 1; 

gamma = 1; gamma_inc = 2;     % eta incremental
for iter = 1: opts.maxIter
    iterTic = tic;
    
    alpha = (t_old  -1 )/t;
    V = W + alpha * (W - W_old);
    
    [ fV , gV] = smoothObjective(V, mu);
    
    for lsIter = 1:100
        % smooth projection
        W = proj(V - gV / gamma, lambda1/gamma);
        [f, W_XWy, W_l2NrmVal] = smoothObjectiveFv(W, mu);
        
        delta_W = W - V;
        r_sum = sum(sum(delta_W.^2));
        
%         if(r_sum <= 1e-20), bFlag = 1; break; end
        
        % line search
        if f<= fV + sum(sum(delta_W .* gV)) + gamma/2 * r_sum
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    lsCnt = lsCnt + lsIter;
    W_old = W;
    
    W_l21NrmVal    =  lambda1 * sum(sqrt(sum(W.^2, 2)));
    funcVal(iter)  = f + W_l21NrmVal; % the smoothed objective function
    info.fvP(iter) = primalObjective(W_XWy, W_l2NrmVal + W_l21NrmVal);
    
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose >= 2
        fprintf('%s: [Iteration %u][fvP %.4g][fvS %.4g][LS %u]\n', ...
            info.algName, iter, info.fvP(iter), funcVal(iter), lsCnt);
    end
    
    % test stop condition.
%     if(bFlag), break; end
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
                if norm(W - opts.W,'fro') <= opts.tol*norm(opts.W,'fro')
                    break;
                end                
        end
    end
    
    mu = mu * muInc;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
end
if verbose == 1; fprintf('\n'); end

%% Output.
info.funcVal = funcVal (1:iter);
info.fvP     = info.fvP(1:iter);
info.timeVal = timeVal (1:iter);

%% Nested Functions
    function fv = primalObjective(XWy, nrmVal)
        % primal objective
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fv =  segL2 (XWy, Th_vecIdx) + nrmVal;
    end

    function [f, g] = smoothObjective(W, mu)
        % f: funcVal of the smooth part of the smoothing function.
        % g: gradient of the smooth part of the smoothing function.
        XWy = Xdiag * W(:) - yvect;
        vv = segL2Proj (XWy/mu, Th_vecIdx);
        f = vv' * XWy - mu/2 * sum(vv.^2) + lambda2 / 2 * sum(sum(W.^2)) ;
        g = reshape( Xdiag' * vv, size(W)) + lambda2 * W;
    end

    function [f, XWy, l2NrmVal] = smoothObjectiveFv(W, mu)
        % f: funcVal of the smooth part of the smoothing function.
        l2NrmVal = lambda2 / 2 * sum(sum(W.^2));
        XWy = Xdiag * W(:) - yvect;
        vv = segL2Proj (XWy/mu, Th_vecIdx);
        f = l2NrmVal + vv' * XWy - mu/2 * sum(vv.^2);
    end
end

function [X] = proj(D, lambda )
% l2.1 norm projection.
X = repmat(max(0, 1 - lambda./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
end
