function [W, info] = MTFLCd_ADMM(X, y, lambda1, lambda2, opts)
%
% Multi-Task Feature Learning with Calibration - ADMM
%
% OBJECTIVE
%    min_W { sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2 }
%
% We use ADMM to solve the problem.
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
opts = setOptsDefault( opts, 'tol',     1e-4);
opts = setOptsDefault( opts, 'stopflag', 1);
verbose = opts.verbose;

info.algName = 'Primal ADMM';

innerOpts = [];
innerOpts = setOptsDefault( innerOpts, 'maxIter', 5000);
innerOpts = setOptsDefault( innerOpts, 'tol',     1e-8);
innerOpts = setOptsDefault( innerOpts, 'tFlag',   1);

if verbose > 0
    fprintf('%s: Config [MaxIter %u][Tol %.4g]\n', info.algName, opts.maxIter, opts.tol);
    fprintf('%s Subsolver: Config [MaxIter %u][Tol %.4g]\n', info.algName, innerOpts.maxIter, innerOpts.tol);
end

m = length(X); % task number
d = size(X{1}, 2); % dimension.
[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);

L2proj = @(x) x ./ max(1, sqrt(sum(x.^2))) ;

% init for variables.
if isfield(opts, 'initTheta')
    Th0 = segL2Proj(opts.initTheta, Th_vecIdx);
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    Th0 = segL2Proj(randn(sum(samplesize), 1), Th_vecIdx);
end

Thvect = zeros(sum(samplesize), 1);
for tt = 1:m
    Thvect(Th_vecIdx(tt)+1:Th_vecIdx(tt+1)) = L2proj(randn(size(X{tt}, 1), 1));
end

Z0 = Th0; % this initialization indicates that w_init = 0.

if isfield(opts, 'initW')
    W = opts.initW;
    if verbose > 0, fprintf('%s: use given initial point.\n', info.algName), end
else
    W = randn(d, m); % starting from a random point.
end

rho = 1; rhoInc = 1;

funcVal  = zeros(opts.maxIter, 1);
timeVal  = zeros(opts.maxIter, 1);
info.fvP = zeros(opts.maxIter, 1);

%% Computation
if verbose == 1; fprintf('Iteration:     '); end

Thvect = Th0;
Zvect  = Z0;
for iter = 1: opts.maxIter
    iterTic = tic;
    
    % Update W
    innerOpts.init = W;
    W = MTFLCd_ADMM_WSolver(Xdiag, yvect, Thvect, Zvect, ...
        rho, lambda1, lambda2, innerOpts); 
%     W2 = MTFLCd_ADMM_WSolver2(Xdiag, yvect, Thvect, Zvect, ...
%         rho, lambda1, lambda2, innerOpts); 
    
    ymXW = yvect - Xdiag * W(:);
    
    % Update Z
    Zvect  = segADMM_Zstep(Thvect/rho + ymXW, Th_vecIdx, rho);
    
    % Update Theta
    Thvect  = Thvect + rho * (ymXW - Zvect);
    
    % Update rho
    rho = rho * rhoInc;
    

    [funcVal(iter), info.fvP(iter)] = augLagObjective(W, Thvect, Zvect);
    
    if iter > 1, timeVal(iter) = timeVal(iter-1) + toc(iterTic);
    else timeVal(iter) = toc(iterTic); end
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    if verbose>1 && iter > 1
            fprintf('%s: [Iter: %u][Fv: %.4f][Fv+L: %.4f](dW %.4g, dZ %.4g, dTh %.4g)\n', ...
                info.algName, iter, info.fvP(iter), funcVal(iter), diffW, diffZ, diffTh);
    end   
    
    % check stop criteria
    if (iter >1)
        switch opts.stopflag
            case 1
                diffW  = sum(sum((W - W_old).^2));
                diffZ  = sum((Zvect  - Zvect_old ).^2);  
                diffTh = sum((Thvect - Thvect_old).^2);

                if( max(diffW, max(diffZ, diffTh)) < opts.tol)
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
    
    W_old  = W; 
    Zvect_old  = Zvect; Thvect_old = Thvect;
end
if verbose == 1; fprintf('\n'); end

%% Output
info.funcVal = funcVal(1:iter);
info.fvP     = funcVal(1:iter);
info.timeVal = timeVal(1:iter);

%% Nested Functions
    function [fvAug, fvP] = augLagObjective(W, Thvect, Zvect)
        % augmented Lagrange objective (fvAug) and original objective (fvP).
        %  
        % fvAug = sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        %         + sum_i^m {Th_i' (y_i - z_i - X_i w_i) + 2 ||y_i - z_i - X_i w_i||^2/rho }
        %
        % fvP  =  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        %         + sum_i^m {Th_i' (y_i - z_i - X_i w_i) + 2 ||y_i - z_i - X_i w_i||^2/rho }
        %
        
        yzXW = yvect - Zvect - Xdiag * W(:);
        fvP  = segL2 (Xdiag * W(:) - yvect, Th_vecIdx) ...
            + lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2));
        fvAug = fvP + Thvect' * yzXW + 2 * sum(yzXW.^2) / rho; 
    end
end



