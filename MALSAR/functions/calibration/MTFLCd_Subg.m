function [ W, info ] = MTFLCd_Subg( X, y, lambda1, lambda2, opts )
%
% Multi-Task Feature Learning with Calibration
% Subgradient descent.
%
% OBJECTIVE
%    min_W { sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
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
% Author: Jiayu Zhou

%% Initialization
if(nargin<5), opts = []; end

opts = setOptsDefault( opts, 'verbose', 1); 
opts = setOptsDefault( opts, 'maxIter', 10000);
opts = setOptsDefault( opts, 'tol',     1e-7);
verbose = opts.verbose;

info.algName = 'Primal Subg';

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', ...
        info.algName, opts.maxIter, opts.tol);
end

m = length(X); % task number
d = size(X{1}, 2);
[Xdiag, ~, Th_vecIdx, yvect] = diagonalize(X, y);

if isfield(opts, 'initW')
    W0 = opts.initW;
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    W0 = randn(d, m); % starting from a random point.
end

funcVal  = zeros(opts.maxIter, 1);
timeVal  = zeros(opts.maxIter, 1);

%% Computation
if verbose == 1; fprintf('Iteration:     '); end
Wk = W0;

[funcVal(1), gWk] = objective(Wk);
timeVal(1) = 0;
for iter = 2: opts.maxIter
    iterTic = tic;
    
    tk = 1/ iter;  %tk = 1/ (lambda2 * iter);
    
    Wk = Wk - tk * gWk; % subgradient step.

    [fWk, gWk] = objective(Wk);
    
    funcVal(iter) = fWk;
    timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose >=2
        fprintf('%s: [Iteration %u][fvP %.4g][none]\n', info.algName, iter, fWk);
    end
    
    if (abs( funcVal(iter) - funcVal(iter-1) ) <=...
            opts.tol* abs(funcVal(iter-1)))
        break;
    end
end
if verbose == 1; fprintf('\n'); end

%% OUTPUT
W   = Wk;
info.funcVal = funcVal(1:iter);
info.fvP     = funcVal(1:iter);
info.timeVal = timeVal(1:iter);

%% NESTED FUNCTION
    function [fv, subg] = objective(W)
        % function value and subgradient of W. 
        XWy = Xdiag * W(:) - yvect;
        
        fv = segL2 (XWy, Th_vecIdx) +...
            lambda1 * L21norm(W) + lambda2 /2 * sum(sum(W.^2));
        
        subg = lambda2 * W + reshape(Xdiag' * segSubg_loss(XWy, Th_vecIdx, eps), size(W))...
                + lambda1 * l2rowSubgrad(W, m);
    end

end

function [Xnrm] = L21norm(X)
% ||X||_{1,2} = sum_i ||X^i||_2
Xnrm = sum(sqrt(sum(X.^2, 2)));
end

function [subg] = l2rowSubgrad(W, m)
    rowl2 = sqrt(sum(W.^2, 2));
    selIdx = abs(rowl2) < eps * 10;
    rowl2 = 1./rowl2;
    rowl2(selIdx) = 0;
    subg = W .* repmat(rowl2, 1, m);
end
