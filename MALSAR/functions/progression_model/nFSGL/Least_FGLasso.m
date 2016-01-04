function [W, funcVal] = Least_FGLasso(X, Y, rho1, rho2, opts)
% Fused Group Lasso with Least Squares Loss.
% By Jiayu Zhou (jiayu.zhou@asu.edu) Jan. 2011

% Objective:
% argmin_W { \sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%              + \sum_{i=1}^d rho1{i} * \|W(i, :)\|_1 
%              + \sum_{i=1}^d rho2{i} \sum_{j=1}^{t-1}* \|W(i, j) - W(i, j+1)\|_1
%          }

% rho1: Lasso sparse parameter.
% rho2: Fused Lasso parameter.

%%% input
% X: {d * n} * t - input matrix
% Y: {n * 1} * t - output matrix
% rho1: d * 1    - weighted Lasso.
% rho2: d * 1    - weighted fused Lasso.

%%% output
% W: model: d * t
% funcVal: function value vector.

if nargin <5
    opts = [];
end

if isfield(opts, 'max_iter')
    max_iter = opts.max_iter;
else
    max_iter = 50;
end

if isfield(opts, 'tol_funcVal')
    tol_funcVal = opts.tol_funcVal;
else
    tol_funcVal = 10^-6;
end

task_num  = length (X);
dimension = size(X{1}, 1);

% if length(rho1) ~= dimension
%     error('Size of rho1 is not correct! Should be equivalent to dimension')
% end
% 
% if length(rho2) ~= dimension
%     error('Size of rho2 is not correct! Should be equivalent to dimension')
% end

funcVal = [];

% Relation
R=zeros(task_num,task_num-1);
R(1:(task_num+1):end)=1;
R(2:(task_num+1):end)=-1;
R = R';

% initial W
W0 = zeros(dimension, task_num);
Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < max_iter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
    
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma, rho2 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        
        Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
            + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1, rho2));
    
    % test stop condition.
    if length(funcVal)> 1 &&...
            (abs(funcVal(end-1) - funcVal(end))/ funcVal(end))<=tol_funcVal
        break;
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [Wp] = FGLasso_projection (W, lambda_1, lambda_2)
        % solve it in row wise, since that
        % \sum_i^d rho1 * \|W(i, :)\|_1 is row coupled.
        % for each row we need to solve the proximal opterator
        % W(i, :) = argmin_w { 0.5 \|w - v\|_2^2
        %            + lambda_1 * \|w\|_1 
        %            + lambda_2 * \|R * w\|_1 } 
        % NOTE: Here the R is t-1 * t, the outside R is t * t-1, and
        Wp = zeros(size(W));
        
        for i = 1 : size(W, 1)
            v = W(i, :);
            
            w0 = zeros(length(v)-1, 1);
            w = flsa(v, w0,  lambda_1, lambda_2, length(v), 1000, 1e-9, 1, 6);
            
            Wp(i, :) = w';
        end
    end

    % smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        grad_W = [];
        for i = 1:task_num
            grad_W = cat(2, grad_W, X{i}*(X{i}' * W(:,i)-Y{i}) );
        end
    end

    % smooth part gradient.
    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
        end
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1, rho_2)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(w, 1) + rho_2 * norm(R * w', 1);
        end
    end

end