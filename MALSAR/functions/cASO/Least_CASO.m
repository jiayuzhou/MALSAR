%% FUNCTION Least_CASO
% Convex-relaxed Alternating Structure Optimization with Least Squares Loss.
%
%% OBJECTIVE
% argmin_{W, M} { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * eta (1+eta) trace(W' (eta I + M)^-1 W)
%     subject to: trace (M) = k, M \preceq I, M \in S_+^d, eta= rho2/rho1
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
%   Logistic_CASO, init_opts

%% Code starts here
function [W, funcVal, M] = Least_CASO(X, Y, rho1, rho2, k, opts)

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

% precomputation.
XX = cell(task_num, 1);
YY = cell(task_num, 1);
XY = cell(task_num, 1);
W0_prep = [];
for t_idx = 1: task_num
    XX{t_idx} = X{t_idx} * X{t_idx}';
    YY{t_idx} = norm(Y{t_idx});
    XY{t_idx} = X{t_idx}*Y{t_idx};
    W0_prep = cat(2, W0_prep, XY{t_idx});
end

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
M0 = speye (dimension) * k / dimension;

bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;
Mz = M0;
Mz_old = M0;

t = 1;
t_old = 0;


iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Ms = (1 + alpha) * Mz - alpha * Mz_old;
    % compute function value and gradients of the search point
    %gWs  = gradVal_eval(Ws, rho1);
    %Fs   = funVal_eval  (Ws, rho1);
    [gWs gMs Fs] = gradVal_eval (Ws, Ms);
    
    while true
        %         [Wzp l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho2 / gamma);
        %         Fzp = funVal_eval  (Wzp, rho1);
        Wzp = Ws - gWs/gamma;
        [Mzp Mzp_Pz Mzp_DiagSigz ] = singular_projection (Ms - gMs/gamma, k);
        Fzp = funVal_eval (Wzp, Mzp_Pz, Mzp_DiagSigz);
        
        %Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        delta_Wzs = Wzp - Ws;
        delta_Mzs = Mzp - Ms;
        
        r_sum = (norm(delta_Wzs, 'fro')^2 + norm(delta_Mzs, 'fro')^2)/2;
        
        
        Fzp_gamma = Fs + sum(sum( delta_Wzs .* gWs)) ...
            + sum(sum( delta_Mzs .* gMs)) ...
            + gamma/2 * norm(delta_Wzs, 'fro')^2 ...
            + gamma/2 * norm(delta_Mzs, 'fro')^2;
        
        
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
    Wz = Wzp;
    Mz_old = Mz;
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

    function [grad_W grad_M funcVal] = gradVal_eval(W ,M)
        invEtaMW = (eta * speye(dimension) + M)\W;
        
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor t_ii = 1:task_num
                XWi = X{t_ii}' * W(:,t_ii);
                XTXWi = X{t_ii}* XWi;
                grad_W(:, t_ii) = XTXWi - XY{t_ii};
            end
        else
            grad_W = [];
            for t_ii = 1:task_num
                XWi = X{t_ii}' * W(:,t_ii);
                XTXWi = X{t_ii}* XWi;
                grad_W = cat(2, grad_W, XTXWi - XY{t_ii});
            end
        end
        grad_W = grad_W + 2 * c * invEtaMW;     %gradient of W component
        grad_M = - c * (invEtaMW * invEtaMW');  %gradient of M component
        
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
        funcVal = funcVal + c * trace( W' * invEtaMW);
    end

    function [funcVal] = funVal_eval (W, M_Pz, M_DiagSigz)
        invIM = M_Pz * ( diag(eta + M_DiagSigz) ) * M_Pz';
        invEtaMW = invIM * W;
        
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        else
            funcVal = 0;
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        end
        funcVal = funcVal + c * trace( W' * invEtaMW);
    end

end