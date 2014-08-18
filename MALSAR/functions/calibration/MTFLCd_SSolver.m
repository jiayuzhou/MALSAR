function [ W, info ] = MTFLCd_SSolver( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration
% diagnoal version for faster computation on small sample size.
% smoothing -- using solver.
%
% OBJECTIVE
%    min_W { sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
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
opts = setOptsDefault( opts, 'epsilon', 1e-1);
verbose = opts.verbose;

info.algName = 'Smth Solver';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', ...
        info.algName, opts.maxIter, opts.tol);
end

m = length(X); % task number
d = size(X{1}, 2);

mu =  opts.epsilon/m; 

if isfield(opts, 'initW')
    W0 = opts.initW;
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    W0 = randn(d, m); % starting from a random point.
end

% diagonalize X and vectorized y.
[Xdiag, ~, Th_vecIdx, yvect] = diagonalize(X, y);

smoothF    = @(W) smoothObjective(W, mu);
non_smooth = l21Proj([d, m], lambda1);

sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'maxfunEv'  , opts.maxIter * 10 ,...
    'maxIter'   , opts.maxIter  ,...
    'ftol'      , opts.tol ,...
    'optim_tol' , opts.tol ,...
    'xtol'      , opts.tol ...
    );

[W, solverInfo] = solver_sparsa( smoothF, non_smooth, W0(:), sparsa_options );

W = reshape(W, d, m);
%info.funcVal = funcVal;
info.fvP     = primalObjective(W);
info.solverInfo = solverInfo;


%% Nested Functions

    function [f, g] = smoothObjective(W, mu)
        % f: funcVal of the smoothing function.
        % g: gradient of the smooth part of the smoothing function.
        g  = lambda2 * W;
        
        XWy = Xdiag * W(:) - yvect;
        vv = segL2Proj (XWy/mu, Th_vecIdx);
        f = lambda2 / 2 * sum(sum(W.^2)) + vv' * XWy - mu/2 * sum(vv.^2);
        
        g = g + reshape( Xdiag' * vv, size(W));
    end

    function fvP = primalObjective(W)
        % primal objective
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fvP = lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2))...
            + segL2 (Xdiag * W(:) - yvect, Th_vecIdx);
    end

end

function op = l21Proj(sizeW, reg_l21)

op = tfocs_prox( @f, @prox_f, [] );

    function v = f(x)
        WMat = reshape(x, sizeW(1), sizeW(2));
        v = reg_l21 * sum(sqrt(sum(WMat.^2, 2)));
    end

    function x = prox_f(x, t)
        WMat = reshape(x, sizeW(1), sizeW(2));
        x = repmat(max(0, 1 - (t * reg_l21)./sqrt(sum(WMat.^2,2))),1,size(WMat,2)).*WMat;
        x = x(:);
    end
end
