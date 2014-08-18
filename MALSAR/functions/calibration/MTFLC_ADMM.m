function [W, info] = MTFLC_ADMM(X, y, lambda1, lambda2, opts)
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
% Author: Jiayu Zhou

%% Initialization
if(nargin<5), opts = []; end

opts = setOptsDefault( opts, 'verbose', 1); 
opts = setOptsDefault( opts, 'maxIter', 10000);
opts = setOptsDefault( opts, 'tol',     1e-4);
verbose = opts.verbose;

innerOpts = [];
innerOpts = setOptsDefault( innerOpts, 'maxIter', 5000);
innerOpts = setOptsDefault( innerOpts, 'tol',     1e-8);
innerOpts = setOptsDefault( innerOpts, 'tFlag',   1);

if verbose > 0
    fprintf('ADMM Config: [MaxIter %u][Tol %.4g]\n', opts.maxIter, opts.tol);
    fprintf('ADMM Subsolver Config: [MaxIter %u][Tol %.4g]\n', innerOpts.maxIter, innerOpts.tol);
end

m = length(X); % task number
d = size(X{1}, 2); % dimension.

funcVal  = zeros(opts.maxIter, 1);

% init for variables.
Th = cell(m, 1); % each element is n_i by 1
for tt = 1:m
    Th{tt} = L2proj(randn(size(X{tt}, 1), 1));
end
Z = Th; % this initialization indicates that w_init = 0.
W = zeros(d, m);

rho = 1; rhoInc = 1;

%% Computation
if verbose == 1; fprintf('Iteration:     '); end
for iter = 1: opts.maxIter
    
    % Update W
    innerOpts.init = W;
    [W, innerInfo] = MTFLC_ADMM_WSolver(X, y, Th, Z, rho, lambda1, lambda2, innerOpts); %#ok
    
    % Update Z (the same size as Tht)
    for t = 1: m
        vt = Th{t} / rho + y{t} - X{t} * W(:, t);
        vtNrm = sqrt(sum(vt.^2));
        Z{t} = max(0, 1- 1/(rho * vtNrm)) * vt;
    end
    
    % Update theta
    for t = 1: m
        Th{t} = Th{t} + rho * (y{t} - Z{t} - X{t} * W(:, t));
    end
    
    %if verbose >=2, fprintf('Th: %.4f\n', augLagObjective(W, Th, Z)); end
    
    % Update rho
    rho = rho * rhoInc;
    
    funcVal(iter) = augLagObjective(W, Th, Z);
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
    
    % check stop criteria
    if (iter >1)
        diffW = sum(sum((W - W_old).^2));
        diffZ = 0; diffTh = 0;
        for t = 1: m
            diffZ  = diffZ  + sum((Z{t}  - Z_old{t}) .^2);
            diffTh = diffTh + sum((Th{t} - Th_old{t}).^2);
        end
        
        if verbose>1
            fprintf('Iter: Fv: %.4f Fv+L: %.4f dW %.4g, dZ %.4g, dTh %.4g\n', ...
                primalObjective(W), funcVal(iter), diffW, diffZ, diffTh);
        end
        
        if( max(diffW, max(diffZ, diffTh)) < opts.tol)
            break;
        end
    end
    
    W_old  = W; Z_old  = Z; Th_old = Th;
end
if verbose == 1; fprintf('\n'); end

%% Output
info.funcVal = funcVal(1: iter);
% use the last to show the 'real' objective without aug Lagrange.
info.funcVal(end + 1) = primalObjective(W);
% NOTE: the primal should be the same as augmented when converged.


%% Nested Functions
    function fvP = primalObjective(W)
        % primal objective (without augmented terms)
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        
        fvP = lambda1 * L21norm(W) + lambda2 /2 * sum(sum(W.^2));
        for i = 1: m
            fvP = fvP + sqrt(sum((X{i} * W(:, i) - y{i}).^2));
        end
    end

    function fvP = augLagObjective(W, Th, Z)
        % primal objective with augmented Lagrange terms.
        %  P(W)  sum_i^m ||Xi wi - yi|| + lambda1 ||W||_{1,2} + lambda2/2 ||W||_F^2
        %         + sum_i^m {Th_i' (y_i - z_i - X_i w_i) + 2 ||y_i - z_i - X_i w_i||^2/rho }
        
        fvP = lambda1 * L21norm(W) + lambda2 /2 * sum(sum(W.^2));
        for i = 1: m
            fvP = fvP + sqrt(sum((X{i} * W(:, i) - y{i}).^2)); % loss
            ti = y{i} - Z{i} - X{i} * W(:, i);
            fvP = fvP + Th{i}' * ti;             % dual
            fvP = fvP + 2 * sum(ti.^2) / rho;    % augment term.
        end
    end

end

function [Xnrm] = L21norm(X)
% ||X||_{1,2} = sum_i ||X^i||_2
Xnrm = sum(sqrt(sum(X.^2, 2)));
end

function [x] = L2proj(x)
nrm = sqrt(sum(x.^2));
x = x ./ max(1, nrm) ;
end

