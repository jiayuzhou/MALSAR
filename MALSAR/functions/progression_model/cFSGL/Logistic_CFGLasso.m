%% FUNCTION Logistic_CFGLasso
%   Fused Group Lasso Progression Model with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}'*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + rho1 * \|W\|_1 + rho2 * \|W*R\|_1  +
%            + rho3 * \|W\|_{2,1} }
%   where:
%   rho1: sparse.
%   rho2: fused Lasso.
%   rho3: L2,1-norm.
%   R encodes fused structure relationship [1 -1 0 ...; 0 1 -1 ...; ...]
%      R=zeros(t,t-1);R(1:(t+1):end)=1;R(2:(t+1):end)=-1;
%
%% INPUT
% X: {d * n} * t - input matrix
% Y: {n * 1} * t - output matrix
% rho1: sparse.
% rho2: fused Lasso.
% rho3: L2,1-norm.
%
%% OUTPUT
% W: model: d * t
% C: model: 1 * t
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
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
%   Least_CFGLasso, init_opts



function [W, C, funcVal] = Logistic_CFGLasso(X, Y, rho1, rho2, rho3, opts)

if nargin <6
    opts = [];
end

% initialize options.
opts=init_opts(opts);

task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];

% precomputation.
R=zeros(task_num,task_num-1);
R(1:(task_num+1):end)=1;
R(2:(task_num+1):end)=-1;
R = R';

%initialize a starting point
C0_prep = zeros(1, task_num);
for t_idx = 1: task_num
    m1 = nnz(Y{t_idx} == 1);
    m2 = nnz(Y{t_idx} == -1);
    if ( m1==0 || m2==0 )
        C0_prep(t_idx) = 0;
    else
        C0_prep(t_idx) = log(m1/m2);
    end
end

if opts.init==2
    W0 = zeros(dimension, task_num);
    C0 = zeros(1, task_num);
elseif opts.init== 0
    W0 = randn(dimension, task_num);
    C0 = C0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0 = zeros(dimension, task_num);
    end
    if isfield(opts,'C0')
        C0=opts.C0;
    else
        C0=C0_prep;
    end
end

bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Cz= C0;
Wz_old = W0;
Cz_old = C0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    
    % compute function value and gradients of the search point
    [gWs, gCs, Fs ]  = gradVal_eval(Ws, Cs);
    
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma, rho2 / gamma, rho3 / gamma);
        Czp = Cs - gCs/gamma;
        Fzp = funVal_eval  (Wzp, Czp);
        
        delta_Wzp = Wzp - Ws;
        delta_Czp = Czp - Cs;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        nrm_delta_Czp = norm(delta_Czp, 'fro')^2;
        r_sum = (nrm_delta_Wzp+nrm_delta_Czp)/2;
        
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + trace(delta_Czp' * gCs)...
        %             + gamma/2 * norm(delta_Wzp, 'fro')^2 ...
        %             + gamma/2 * norm(delta_Czp, 'fro')^2;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs))...
            + sum(sum(delta_Czp .* gCs))...
            + gamma/2 * nrm_delta_Wzp ...
            + gamma/2 * nrm_delta_Czp;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Cz_old = Cz;
    Wz = Wzp;
    Cz = Czp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1, rho2, rho3));
    
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
C = Czp;

% private functions

    function [Wp] = FGLasso_projection (W, lambda_1, lambda_2, lambda_3 )
        % solve it in row wise (L_{2,1} is row coupled).
        % for each row we need to solve the proximal opterator
        % argmin_w { 0.5 \|w - v\|_2^2
        %            + lambda_1 * \|w\|_1 + lambda_2 * \|R * w\|_1  +
        %            + lambda_3 * \|w\|_2 }
        % NOTE: Here the R is t-1 * t, the outside R is t * t-1, and
        
        Wp = zeros(size(W));
        
        for i = 1 : size(W, 1)
            v = W(i, :);
            w = FGLasso_projection_rowise(v, lambda_1, lambda_2, lambda_3);
            Wp(i, :) = w';
        end
    end

% smooth part gradient and function value. .
    function [grad_W, grad_C, funcVal] = gradVal_eval(W, C)
        grad_W = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);
        if opts.pFlag
            parfor i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i} );
            end
        else
            for i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i} );
            end
        end
        %grad_W = grad_W+ rho1 * 2 *  W * RRt;
        funcVal = sum(lossValVect);% + rho1 * norm(W*R, 'fro')^2;
    end

% smooth part function value.
    function [funcVal] = funVal_eval (W, C)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            for i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end
        %funcVal = funcVal + rho1 * norm(W*R, 'fro')^2;
    end

% non-smooth part function value.
    function [non_smooth_value] = nonsmooth_eval(W, rho_1, rho_2, rho_3)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(w, 1) + rho_2 * norm(R * w', 1) ...
                + rho_3 * norm(w, 2);
        end
    end
end

function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y )
%Logistic_Eval. evaluation of logistic loss.
m = length(y);
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_c = sum(b);
grad_w = x * b;
end

function [ funcVal ] = unit_funcVal_eval( w, c, x, y)
%function value evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end