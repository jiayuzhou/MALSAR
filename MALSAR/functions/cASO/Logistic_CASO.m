%% FUNCTION Logistic_CASO
%   Convex-relaxed Alternating Structure Optimization with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,M,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + rho1 * eta (1+eta) trace(W' (eta I + M)^-1 W)
%     subject to: trace (M) = k, M \preceq I, M \in S_+^d, eta = rho2/rho1
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   k: dimension of shared structure
%   rho1: task relatedness controlling parameter (rho1=0 then tasks are not
%   related)
%   rho2: L2 norm regularization on model W (rho2=0 then reduce to ASO but
%   this solver does not support this case)
%
%% OUTPUT
%   W: model: d * t
%   C: model: 1 * t
%   funcVal: function value vector.
%   M: relaxed Theta' * Theta , where Theta is the shared subspace
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jianhui Chen and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 22, 2012.
%
%% RELATED PAPERS
%
%   [1] J. Chen, L. Tang, J. Liu, and J. Ye. A Convex Formulation
%   for Learning Shared Structures from Multiple Tasks. ICML 2009.
%
%% RELATED FUNCTIONS
%   Least_CASO, init_opts

%% Code starts here
function [W, C, funcVal, M] = Logistic_CASO(X, Y, rho1, rho2, k, opts)

if nargin <5
    error('\n Inputs: X, Y, rho1, rho2 and k should be specified!\n');
end
X = multi_transpose(X);

if nargin <6
    opts = [];
end

if rho2<=0 || rho1<=0
    error('rho1 and rho2 should both greater than zero.');
end

% if exist('mosekopt','file')==0
%     error('Mosek is not found. Please install Mosek first. \n')
% end

% initialize options.
opts=init_opts(opts);

task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];

eta = rho2 / rho1;
c = rho1 * eta * (1 + eta);

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

M0 = speye (dimension) * k / dimension;


bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Cz= C0;
Mz = M0;

Wz_old = W0;
Cz_old = C0;
Mz_old = M0;

t = 1;
t_old = 0;
iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    Ms = (1 + alpha) * Mz - alpha * Mz_old;
    
    % compute function value and gradients of the search point
    [gWs, gCs, gMs, Fs ]  = gradVal_eval(Ws, Cs, Ms);
    
    % the Armijo Goldstein line search scheme
    while true
        %[Wzp l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho2 / gamma);
        %Fzp = funVal_eval  (Wzp, Czp, rho1);
        
        Wzp = Ws - gWs/gamma;
        Czp = Cs - gCs/gamma;
        [Mzp Mzp_Pz Mzp_DiagSigz ] = singular_projection (Ms - gMs/gamma, k);
        Fzp = funVal_eval (Wzp, Czp, Mzp_Pz, Mzp_DiagSigz);
        
        delta_Wzp = Wzp - Ws;
        delta_Czp = Czp - Cs;
        delta_Mzp = Mzp - Ms;
        
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        nrm_delta_Czp = norm(delta_Czp, 'fro')^2;
        nrm_delta_Mzp = norm(delta_Mzp, 'fro')^2;
        
        r_sum = (nrm_delta_Wzp+nrm_delta_Czp+nrm_delta_Mzp)/3;
        
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) ...
            + sum(sum(delta_Czp .* gCs))...
            + sum(sum(delta_Mzp .* gMs)) ...
            + gamma/2 * nrm_delta_Wzp ...
            + gamma/2 * nrm_delta_Mzp ...
            + gamma/2 * nrm_delta_Czp;
        
        if (r_sum <=eps)
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
    Mz_old = Mz;
    
    Wz = Wzp;
    Cz = Czp;
    Mz = Mzp;
    
    funcVal = cat(1, funcVal, Fzp);
    
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
M = Mzp;

% private functions

    function [Mzp Mzp_Pz Mzp_DiagSigz ] = singular_projection (Msp, k)
        [EVector EValue] = eig(Msp);
        Pz = real(EVector);  diag_EValue = real(diag(EValue));
        %DiagSigz = SingVal_Projection(diag_EValue, k);
        DiagSigz = bsa_ihb(diag_EValue, ones(size(diag_EValue)), k, ones(size(diag_EValue)));
        Mzp = Pz * diag(DiagSigz) *Pz';
        Mzp_Pz = Pz;
        Mzp_DiagSigz = DiagSigz;
    end


    function [grad_W, grad_C, grad_M, funcVal] = gradVal_eval(W, C, M)
        invEtaMW = (eta * speye(dimension) + M)\W;
        
        grad_W = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);
        
        if opts.pFlag
            parfor i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            for i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end
        grad_W = grad_W + 2 * c * invEtaMW;     %gradient of W component
        grad_M = - c * (invEtaMW * invEtaMW');  %gradient of M component
        
        funcVal = sum(lossValVect) + c * trace( W' * invEtaMW);
    end



    function [funcVal] = funVal_eval (W, C, M_Pz, M_DiagSigz)
        invIM = M_Pz * ( diag(eta + M_DiagSigz) ) * M_Pz';
        invEtaMW = invIM * W;
        
        if opts.pFlag
            funcVal = 0;
            parfor i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            funcVal = 0;
            for i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end
        funcVal = funcVal  + c * trace( W' * invEtaMW);
    end


end


function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y)
%gradient and logistic evaluation for each task
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