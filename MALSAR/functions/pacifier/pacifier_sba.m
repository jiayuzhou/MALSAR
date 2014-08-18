function [ U, V, Ss, fv ] = ...
    pacifier_sba( X0, rank_max, reg_l1, reg_l2, reg_smooth, opts)
%PERFACT PatiEnt analysis via low-Rank FACtorization.
%
% Input:
%    X0: cell of sparse matrices, .
%    Omega: cell
%
if nargin<6, opts = []; end
verbose = 2;

fv = [];
maxIter = 100;
tol = 1e-5;
estRank = true;

if isfield(opts, 'maxIter'), maxIter = opts.maxIter; end
if isfield(opts, 'tol'),     tol     = opts.tol;     end
if isfield(opts, 'estRank'), estRank = opts.estRank; end
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
U0 = zeros(d, r);
V0 = cell(n, 1);

X0Arr = cell(n, 1); % the array storage of X0.
% TODO: replace the original X0.

iR = cell(n, 1); iC = cell(n, 1);
Ss= cell(n, 1);
for nn = 1: n
    
    % WARNING [move to data generation].
    %data(data == 0) = eps; % so the known 0s will not be considered as sparse element
    %[iR{nn}, iC{nn}] = ind2sub([d k], Known{nn}); %S_comp = sparse(iR{nn}, iC{nn}, data, m, n);
    
    ttt = sparse(full(X0{nn}));
    V0{nn} = randn(r, size(ttt, 2));
    [iR{nn}, iC{nn}, X0Arr{nn}] = find(ttt); % this sort the indices in iR and iC.
    Ss{nn} = ttt;% !!! MATLAB BUG BY PASSING REFERENCE OF SPARSE MATRICES
    
    %V0{nn} = randn(r, size(X0{nn}, 2));
    %[iR{nn}, iC{nn}, X0Arr{nn}] = find(X0{nn}); % this sort the indices in iR and iC.
    %Ss{nn} = X0{nn};
end

%% rank estimation

itr_rank = 0; % this will be reset after every reset.
minitr_reduce_rank = 4;  % itr_rank > minitr_reduce_rank
maxitr_reduce_rank = 5; % itr_rank > maxitr_reduce_rank, eval rank occurs
rank_min =  1;
rk_jump = 10;


%% Main iterations
if verbose == 1; fprintf('Iteration:     '); end
for iter = 1: maxIter
    itr_rank = itr_rank + 1;

    if verbose == 2, fprintf('Iteration %u (Max Iter: %u)...\n', iter, maxIter);end
    
    % %% Solve U  given Vi and Si
    %    optimization problem that solves U.
    U = solveU(U0, V0, Ss, reg_l1, l1_solver_options);
    % FINAL: add some options to allow regularization change.
    
    % ERROR: computation issue: functional value increases.
    
    % %% Solve Vi given U and Si
    V = solveV_eig(U, U0, V0, Ss, reg_smooth, reg_l2);
    % NOTE: use U instead of U0 accelerates computation. Gaussi Siedal.
    
    % %% Solve Si given U and Vi
    %    matrix completion step
    %    % S_comp = X0 - pomg_UV;
    for i = 1: length(Ss)
        s_comp_val = X0Arr{i} - sparse_inp(U', V{i}, iR{i}, iC{i})';
        sparse_update(Ss{i}, s_comp_val);
    end
    
    fv(end+1) = evalFuncVal(U, V, U, V, Ss, reg_l1, reg_l2, reg_smooth); %#ok
    if verbose == 2, fprintf('function val at iter [%u]: %.4g\n', iter, fv(end)); end
    
    % convergence test.
    diffNm = maxNorm(V, V0, U, U0);
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
    
    % rank estimator
    if estRank >= 1 && itr_rank > minitr_reduce_rank
        R = qr(U, 0);
        rk = rank_estimator_adaptive(R, r, itr_rank, rk_jump, rank_min, minitr_reduce_rank, maxitr_reduce_rank);
        if rk ~= r;
            % if rank changed
            if verbose ==2, fprintf('[Rank Estimation] Rank changed from %u to %u\n', r, rk); end
            r = rk;
            U = U(:,1:rk);
            for i = 1: length(Ss)
                V{i} =  V{i}(1:rk,:);
                s_comp_val = X0Arr{i} - sparse_inp(U', V{i}, iR{i}, iC{i})';
                sparse_update(Ss{i}, s_comp_val);
            end
            itr_rank = 0; % reset iteration count for rank. 
            estRank = 0; % run estimation only once.
            % update function value. 
            fv(end) = evalFuncVal(U, V, U, V, Ss, reg_l1, reg_l2, reg_smooth); 
        end
    end
    
    % update variables.
    V0 = V;
    U0 = U;
    
    if verbose == 1; fprintf('\b\b\b\b\b%5i',iter); end
end
if verbose ==1; fprintf(' Done.\n'); end



end


function [fv, fvll1] = evalFuncVal(U, V, Us, Vs, Ss, reg_l1, reg_l2, reg_smooth)

funcVal_unit =@(A, B, P, U, V) ...
    sum(sum(B.*((A' * A) * B))) + 2 * sum(sum((A' * P).* B)) ...
    + sum(sum(P.* P)) - 2 * sum(sum(B .* ((A' * U) * V))) ...
    - 2* sum(sum((U' * P) .* V)) + sum(sum(V .* ((U' * U) * V)));

fv = 0;
fvll1 = 0; %loss and l1.
for i = 1: length(V)
    ti = size(Ss{i}, 2);
    iy  = [1: (ti - 1), 1: (ti - 1)];
    ix  = [1: (ti - 1), 2: ti];
    val = [ones(1, ti-1), -1 * ones(1, ti-1)];
    Ri  = sparse(ix, iy, val, ti, ti-1);
    
    lossVal = funcVal_unit(Us, Vs{i}, Ss{i}, U, V{i})/2/ti;
    fv = fv + lossVal;
    fvll1 = fvll1 + lossVal;
    
    fv = fv + reg_l2 * norm(V{i}, 'fro')^2 /2/ti;
    fv = fv + reg_smooth * norm(V{i} * Ri, 'fro')^2 /2/ti;
end
l1u = reg_l1 * sum(abs(U(:)));
fv = fv + l1u;
fvll1 = fvll1 + l1u;

end

function est_rk = rank_estimator_adaptive(R, k, ...
    itr_rank, rk_jump, rank_min, minitr_reduce_rank, maxitr_reduce_rank)

dR = abs(diag(R));       drops = dR(1:end-1)./dR(2:end);
[dmx,imx] = max(drops);  rel_drp = (k-1)*dmx/(sum(drops)-dmx);

if (rel_drp > rk_jump && itr_rank > minitr_reduce_rank) ...
        || itr_rank > maxitr_reduce_rank; %bar(drops); pause;
    est_rk = max([imx, floor(0.1*k), rank_min]);
else
    est_rk = k;
end
end