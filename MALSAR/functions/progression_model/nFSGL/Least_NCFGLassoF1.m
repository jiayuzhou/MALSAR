function [W, funcVal] = Least_NCFGLassoF1(X, Y, lambda, gamma, opts)
% Non-Convex Fused Group Lasso with Least Squares Loss (Formulation 1). 
% By Jiayu Zhou (jiayu.zhou@asu.edu) Dec. 2011

% Objective (Non-Convex):
% argmin_W { \sum_{i=1}^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%              + lambda * \sum_{i=1}^d \sqrt{\|W(i, :)\|_1}
%              + gamma  * \|W*R\|_1
%          }
%
% t - task number 
% d - dimension
% n - sample size
%
% lambda: L(1,0.5)-norm. joint feature selection
% gamma:   fused Lasso.
% R encodes fused structure relationship [1 -1 0 ...; 0 1 -1 ...; ...]
%    R=zeros(t,t-1);R(1:(t+1):end)=1;R(2:(t+1):end)=-1;

%%% input
% X: {d * n} * t - input matrix
% Y: {n * 1} * t - output matrix
% lambda: 1        - L_{1, 0.5}-norm sparsity. joint feature selection
% gamma: 1        - fused Lasso.

if nargin <5
    opts = [];
end

if isfield(opts, 'max_iter')
    max_iter = opts.max_iter;
else
    max_iter = 20;
end

if isfield(opts, 'tol_funcVal')
    tol_funcVal = opts.tol_funcVal;
else
    tol_funcVal = 10^-6;
end

task_num  = length (X);
dimension = size(X{1}, 1);

% Relation
R=zeros(task_num,task_num-1);
R(1:(task_num+1):end)=1;
R(2:(task_num+1):end)=-1;
R = R';


if isfield(opts, 'starting_point')
    %fprintf('Least_NCFGLassoF1: Init model used\n');
    W0 = opts.starting_point;
else
    %W0 = rand(dimension, task_num);
    [W0] = Least_FGLasso(X, Y, lambda, gamma);
end

epsilon = 10e-10;

funcVal = funcVal_eval (W0);

inner_iteration_opts.max_iter = 100;
inner_iteration_opts.tol_funcVal = 10^-7;

iter = 0;
while iter < max_iter
    % reweigting vector.
    
    rho1 = zeros(dimension, 1);
    for t = 1: dimension
        rho1(t) = lambda / (2 * sqrt(norm(W0(t, :),1) + epsilon));
    end
    rho2 = gamma;
    
    %[W, sub_funcVal] = Least_Weight1FGLasso(X, Y, rho1, rho2, inner_iteration_opts);
    [W] = Least_Weight1FGLasso(X, Y, rho1, rho2, inner_iteration_opts);
    
    funcVal = cat(1, funcVal, funcVal_eval (W));
    
    % test stop condition.
    if length(funcVal)> 1 &&...
            (abs(funcVal(end-1) - funcVal(end))/ funcVal(end))<=tol_funcVal
        break;
    end
    
    iter = iter + 1;
    W0 = W;
    
end

    function [funcVal] = funcVal_eval (W)
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
        end
        % non-smooth part
        for i = 1 : size(W, 1) % dimension 
            w = W(i, :);
            funcVal = funcVal ...
                + lambda * sqrt(norm(w, 1)) + gamma * norm(R * w', 1);
        end
    end

end