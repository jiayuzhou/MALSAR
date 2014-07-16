%% FUNCTION Least_Weight2FGLasso
%   Weighted Fused Group Lasso and Weighted Lasso with Least Squares Loss .
%
%% OBJECTIVE
%   argmin_W { \sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%              + \sum_{i=1}^d rho1{i} * \|W(i, :)\|_1
%              + \sum_{i=1}^d rho2{i} \sum_{j=1}^{t-1}* \|W(i, j) - W(i, j+1)\|_1
%          }
%   rho1: weighted Lasso sparse.
%   rho2: weighted Fused Lasso.
%   R encodes fused structure relationship [1 -1 0 ...; 0 1 -1 ...; ...]
%      R=zeros(t,t-1);R(1:(t+1):end)=1;R(2:(t+1):end)=-1;
%
%% INPUT
% X: {d * n} * t - input matrix
% Y: {n * 1} * t - output matrix
% rho1: d * 1    - weighted Lasso.
% rho2: d * 1    - weighted fused Lasso.
%
%% OUTPUT
% W: model: d * t
% funcVal: function value vector.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jun Liu and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% RELATED PAPERS
%
%   [1] Zhou, J., Jun, L., Narayan, A. V. and Ye, J.  Modeling Disease
%   Progression via Fused Sparse Group Lasso. KDD 2012
%
%% RELATED FUNCTIONS
%   Least_NCFGLasso, init_opts

function [W, funcVal] = Least_Weight2FGLasso(X, Y, rho1, rho2, opts)


if nargin <5
    opts = [];
end

% initialize options.
opts=init_opts(opts);


task_num  = length (X);
dimension = size(X{1}, 1);

if length(rho1) ~= dimension
    error('Size of rho1 is not correct! Should be equivalent to dimension')
end

if length(rho2) ~= dimension
    error('Size of rho2 is not correct! Should be equivalent to dimension')
end

funcVal = [];

% Relation
R=zeros(task_num,task_num-1);
R(1:(task_num+1):end)=1;
R(2:(task_num+1):end)=-1;
R = R';

% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end

bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
    
    while true
        Wzp = ReweightFGLasso_projection(Ws - gWs/gamma, rho1 ./ gamma, rho2 ./ gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp.* gWs))...
            + gamma/2 * r_sum;
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1, rho2));
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [Wp] = ReweightFGLasso_projection (W, lambda_1, lambda_2)
        % solve it in row wise, since that
        % \sum_i^d rho1{i} * \|W(i, :)\|_1 is row coupled.
        % for each row we need to solve the proximal opterator
        % W(i, :) = argmin_w { 0.5 \|w - v\|_2^2
        %            + lambda_1{i} * \|w\|_1
        %            + lambda_2{i} * \|R * w\|_1 }
        % NOTE: Here the R is t-1 * t, the outside R is t * t-1, and
        Wp = zeros(size(W));
        
        for i = 1 : size(W, 1)
            v = W(i, :);
            
            w0 = zeros(length(v)-1, 1);
            w = flsa(v, w0,  lambda_1(i), lambda_2(i), length(v), 1000, 1e-9, 1, 6);
            
            Wp(i, :) = w';
        end
    end

% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor i = 1:task_num
                grad_W(:, i) = X{i}*(X{i}' * W(:,i)-Y{i});
            end
        else
            grad_W = [];
            for i = 1:task_num
                grad_W = cat(2, grad_W, X{i}*(X{i}' * W(:,i)-Y{i}) );
            end
        end
    end

% smooth part gradient.
    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        end
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1, rho_2)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1(i) * norm(w, 1) + rho_2(i) * norm(R * w', 1);
        end
    end

end