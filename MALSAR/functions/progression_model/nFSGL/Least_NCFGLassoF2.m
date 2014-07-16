function [W, funcVal] = Least_NCFGLassoF2(X, Y, lambda, beta, opts)
% Non-Convex Fused Group Lasso with Least Squares Loss (Formulation 2). 
% By Jiayu Zhou (jiayu.zhou@asu.edu) Dec. 2011

% Objective (Non-Convex):
% argmin_W { \sum_{i=1}^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%    + lambda * \sum_{i=1}^d \sqrt{ \|R * W(i, :)\|_1 + beta  * \|W(i, :)\|_1}
%          }
%
% t - task number 
% d - dimension
% n - sample size
%
% lambda: joint feature selection
% beta:   Lasso sparsity.
% R encodes fused structure relationship [1 -1 0 ...; 0 1 -1 ...; ...]
%    R=zeros(t,t-1);R(1:(t+1):end)=1;R(2:(t+1):end)=-1;

%%% input
% X: {d * n} * t - input matrix
% Y: {n * 1} * t - output matrix
% lambda: 1        - joint feature selection
% beta:   1        - Lasso sparsity.

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
    %fprintf('Least_NCFGLassoF2: Init model used\n');
    W0 = opts.starting_point;
else
    %W0 = rand(dimension, task_num);
    [W0] = Least_FGLasso(X, Y, lambda*beta, lambda);
end

epsilon = 10e-12; % should be slight larger than machine error.

funcVal = funcVal_eval (W0);

inner_iteration_opts.max_iter = 100;
inner_iteration_opts.tol_funcVal = 10^-7;

iter = 0;
while iter < max_iter
    % reweigting vector.
    
    rho1 = zeros(dimension, 1); % weighted Lasso 
    rho2 = zeros(dimension, 1); % weighted Fused Lasso
    for t = 1: dimension
        rho2(t) = lambda / (2 * sqrt(norm(W0(t, :),1) + norm(R * W0(t, :)', 1) + epsilon));
        rho1(t) = rho2(t) * beta; 
    end
    
    [W, sub_funcVal] = Least_Weight2FGLasso(X, Y, rho1, rho2, inner_iteration_opts);
    
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
                + lambda * sqrt(norm(R * w', 1) + beta * norm(w, 1));
        end
    end

end