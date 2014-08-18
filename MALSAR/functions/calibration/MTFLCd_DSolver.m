function [ W, info, Thstar ] = MTFLCd_DSolver( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration
% diagnoal version for faster computation on small sample size.
% dual projected gradient -- using solver. 
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
verbose = opts.verbose;

info.algName = 'Dual Solver';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', info.algName, opts.maxIter, opts.tol);
end

m = length(X); % task number
d = size(X{1}, 2);

% diagonalize X and vectorized y.
[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);


if isfield(opts, 'initTheta')
    Th0 = segL2Proj(opts.initTheta, Th_vecIdx);
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    Th0 = segL2Proj(randn(sum(samplesize), 1), Th_vecIdx);
end


smoothF    = @(th) dualSmoothNeg(th);
non_smooth = segProjection();

sparsa_options = pnopt_optimset(...
          'display'   , 0    ,...
          'maxfunEv'  , opts.maxIter * 10 ,...
          'maxIter'   , opts.maxIter  ,...
          'ftol'      , opts.tol ,...
          'optim_tol' , opts.tol ,...
          'xtol'      , opts.tol ...
         );

Thstar = solver_sparsa( smoothF, non_smooth, Th0, sparsa_options );

W = computeW(Thstar);
info.funcVal = primalObjective(W);
info.fvP     = info.funcVal;
info.fvD     = dualSmooth(Thstar);


%% Nested Functions
    function fvP = primalObjective(W)
        % primal objective
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        fvP = lambda1 * L21norm(W) + lambda2 /2 * sum(sum(W.^2))...
            + segL2 (Xdiag * W(:) - yvect, Th_vecIdx);
    end

    function WTh = computeW(Th_vec)
        WTh = reshape(Th_vec' * Xdiag, d, m);% Compute UTh
        WTh = -1/lambda2*max(0,1-lambda1./repmat(sqrt(sum(WTh.^2,2)),1,m)).*WTh;
    end

    function [f, g, WTh] = dualSmooth(Th_vec)
        WTh = computeW(Th_vec); % compute the corresponding W(Theta).
        % gradient 
        g = Xdiag * WTh(:) - yvect;
        % function value.
        f = lambda1 * L21norm(WTh) + lambda2 / 2 * sum(sum(WTh.^2))...
            + Th_vec' * g;
    end

    function [f, g] = dualSmoothNeg(Th_vec)
        [f, g] = dualSmooth(Th_vec);
        f = -f;
        g = -g;
    end

    function op = segProjection()
        op = @(varargin)segProjWrapper( varargin{:} );
    end

    function [ v, x ] = segProjWrapper( x, t ) %#ok
    v = 0; % ignore. 
    switch nargin,
        case 1,
            %do nothing.
        case 2,
            x = segL2Proj(x, Th_vecIdx);
        otherwise,
            error( 'Not enough arguments.' );
    end
    
    end

end

function [l21Val] = L21norm(X)
    l21Val= sum(sqrt(sum(X.^2, 2))); % L2,1 norm computation. 
end