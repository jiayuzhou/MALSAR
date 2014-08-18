function [ U, V, Ss, fv ] = pacifier_iba( X0, rank_max, reg_l1, reg_l2, reg_smooth, opts)
%PERFACT PatiEnt analysis via low-Rank FACtorization.
%
% Input:
%    X0: cell of sparse matrices, .
%    Omega: cell
%
if nargin<6, opts = []; end
verbose = 1;

fv = [];
maxIter = 10;
tol = 1e-5;

if isfield(opts, 'maxIter'), maxIter = opts.maxIter; end
if isfield(opts, 'tol'),     tol     = opts.tol;     end
if isfield(opts, 'verbose'), verbose = opts.verbose; end

l1_solver_options = pnopt_optimset(...
    'display'   , 0  ,...
    'maxfunEv'  , 5000 ,...
    'maxIter'   , 500  ,...
    'ftol'      , 1e-9 ,...
    'optim_tol' , 1e-6 ,...
    'xtol'      , 1e-9 ...
    );

d = size(X0{1},1);
n = length(X0);
r = rank_max;

%% initialization
U0 = cell(n, 1);
V0 = cell(n, 1);

X0Arr = cell(n, 1); % the array storage of X0.
% TODO: replace the original X0.

iR = cell(n, 1); iC = cell(n, 1);
Ss= cell(n, 1);
for nn = 1: n
    U0{nn} = zeros(d, r);
    V0{nn} = randn(r, size(X0{nn}, 2));
    ttt = sparse(full(X0{nn}));
    [iR{nn}, iC{nn}, X0Arr{nn}] = find(ttt); % this sort the indices in iR and iC.
    Ss{nn} = ttt; % !!! MATLAB BUG BY PASSING REFERENCE OF SPARSE MATRICES
    %[iR{nn}, iC{nn}, X0Arr{nn}] = find(X0{nn}); % this sort the indices in iR and iC.
    %Ss{nn} = X0{nn};
end



%% Main iterations
if verbose == 1; fprintf('Iteration:     '); end
for iter = 1: maxIter
    %itr_rank = itr_rank + 1;

    if verbose == 2, fprintf('Iteration %u (Max Iter: %u)...\n', iter, maxIter);end
    
    % %% Solve U  given Vi and Si
    %    optimization problem that solves U.
    U = solveU_ind(U0, V0, Ss, reg_l1, l1_solver_options);
    
    % %% Solve Vi given U and Si
    V = solveV_eig_ind(U, U0, V0, Ss, reg_smooth, reg_l2);
    % NOTE: use U instead of U0 accelerates computation. Gaussi Siedal.
    
    % %% Solve Si given U and Vi
    %    matrix completion step
    %    % S_comp = X0 - pomg_UV;
    for i = 1: length(Ss)
        s_comp_val = X0Arr{i} - sparse_inp(U{i}', V{i}, iR{i}, iC{i})';
        sparse_update(Ss{i}, s_comp_val);
    end
    
    fv(end+1) = evalFuncVal_ind(U, V, U, V, Ss, reg_l1, reg_l2, reg_smooth); %#ok
    if verbose == 2, fprintf('function val at iter [%u]: %.4g\n', iter, fv(end)); end
    
    % convergence test.
    diffNm = maxNorm_ind(V, V0, U, U0);
    if diffNm<tol
        if verbose ==2, fprintf('max norm tol reached [%.4g | tol: %.4g].\n', diffNm, tol); end
        break;
    end
    if length(fv) >1
        relDiffFv = (fv(end-1) - fv(end))/fv(end-1);
        if relDiffFv < tol
            if verbose ==2, fprintf('relative fv tol reached [%.4g | tol: %.4g].\n', relDiffFv, tol); end
            break;
        end
    end
    
    % update variables.
    V0 = V;
    U0 = U;
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
end
if verbose ==1; fprintf(' Done.\n'); end



end


function [fv] = evalFuncVal_ind(U, V, Us, Vs, Ss, reg_l1, reg_l2, reg_smooth)

funcVal_unit =@(A, B, P, U, V) ...
    sum(sum(B.*((A' * A) * B))) + 2 * sum(sum((A' * P).* B)) ...
    + sum(sum(P.* P)) - 2 * sum(sum(B .* ((A' * U) * V))) ...
    - 2* sum(sum((U' * P) .* V)) + sum(sum(V .* ((U' * U) * V)));

fv = 0;
l1u = 0;
for i = 1: length(V)
    ti = size(Ss{i}, 2);
    iy  = [1: (ti - 1), 1: (ti - 1)];
    ix  = [1: (ti - 1), 2: ti];
    val = [ones(1, ti-1), -1 * ones(1, ti-1)];
    Ri  = sparse(ix, iy, val, ti, ti-1);
    
    lossVal = funcVal_unit(Us{i}, Vs{i}, Ss{i}, U{i}, V{i})/2/ti;
    fv = fv + lossVal;
    
    l1u = l1u + reg_l1 * sum(abs(U{i}(:))); 
    
    fv = fv + reg_l2 * norm(V{i}, 'fro')^2 /2/ti;
    fv = fv + reg_smooth * norm(V{i} * Ri, 'fro')^2 /2/ti;
end

fv = fv + l1u;


end
