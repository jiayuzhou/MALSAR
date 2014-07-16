%% FUNCTION Least_Dirty
%   Dirty Multi-Task Learning with Least Squares Loss.
%
%% OBJECTIVE
%   argmin_W ||X(P+Q) - Y||_F^2 + lambda1*||P||_{1,inf} + lambda2*||Q||_{1,1}
%    s.t. W = P + Q
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: group sparsity regularization parameter
%   rho2: elementwise sparsity regularization parameter
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
%   P: group sparsity structure (joint feature slection)
%   Q: elementwise sparsity component
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Pinghua Gong and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% RELATED PAPERS
%
%   [1] Jalali, A. and Ravikumar, P. and Sanghavi, S. and Ruan, C. A dirty
%       model for multi-task learning, NIPS 2010.
%
%% RELATED FUNCTIONS
%   init_opts, combine_input (utils), prf_lbm (c_files)

function [W, funcVal, P, Q] = Least_Dirty(X, Y, lambda1, lambda2, opts)

if nargin <4
    error('\n Inputs: X, Y, and lambda1, and lambda2 should be specified!\n');
end
if nargin <5
    opts = [];
end

% initialize options.
opts=init_opts(opts);

% initial Lipschiz constant. 
if isfield(opts, 'lFlag')
    lFlag = opts.lFlag;
else
    lFlag = false;
end


task_num = length(X);
[X, y, ~, samplesize] = combine_input(X, Y);
dimension = size(X, 2);

% initialize a starting point
if opts.init==2
    P0 = zeros(dimension, task_num);
    Q0 = zeros(dimension, task_num);
elseif opts.init == 0
    P0 = randn(dimension, task_num);
    Q0 = randn(dimension, task_num);
else
    if isfield(opts,'P0')
        P0=opts.P0;
        if (nnz(size(P0)-[dimension, task_num]))
            error('\n Check the input .P0');
        end
    else
        P0=zeros(dimension, task_num);
    end
    
    if isfield(opts,'Q0')
        Q0=opts.Q0;
        if (nnz(size(Q0)-[dimension, task_num]))
            error('\n Check the input .Q0');
        end
    else
        Q0=zeros(dimension, task_num);
    end
end

% Set an array to save the objective value
funcVal = [];


P = P0; Q = Q0;

%[d,m] = size(P); % d: dimension, m: the number of tasks
X = diagonalize(X,samplesize);
XtX = X'*X; Xty = X'*y;

Pn = P;  Qn = Q;
t_new = 1;
L1norm = max(sum(abs(X),1)); Linfnorm = max(sum(abs(X),2));

if lFlag
    % Upper bound for largest eigenvalue of Hessian matrix
    L = 2*min([L1norm*Linfnorm; size(X,1)*Linfnorm*Linfnorm; size(X,2)*L1norm*L1norm; size(X,1)*size(X,2)*max(abs(X(:)))]);
else
    % Lower bound for largest eigenvalue of Hessian matrix
    L = 2*max(L1norm*L1norm/size(X,1),Linfnorm*Linfnorm/size(X,2));
end
% Initial function value
funcVal = cat(1, funcVal, norm(X*(P(:) + Q(:)) - y)^2 + lambda1*L1infnorm(P) + lambda2*L11norm(Q));
%fun(1) = norm(X*(P(:) + Q(:)) - y)^2 + lambda1*L1infnorm(P) + lambda2*L11norm(Q);

%count = 0;
for iter = 1 : opts.maxIter
    P_old = P; Q_old = Q;
    t_old = t_new;
    gradvec = 2*(XtX*(Pn(:)+Qn(:)) - Xty);
    gradmat = reshape(gradvec,dimension,task_num);
    % If we estimate the upper bound of Lipschitz constant, no line search
    % is needed.
    if lFlag
        P = proximalL1infnorm(Pn - gradmat/L, lambda1/L);
        Q = proximalL11norm(Qn - gradmat/L, lambda2/L);
    else
        % line search
        for inneriter = 1:20
            P = proximalL1infnorm(Pn - gradmat/L, lambda1/L);
            Q = proximalL11norm(Qn - gradmat/L, lambda2/L);
            dP = P - Pn;  dQ = Q - Qn;
            if 2*((dP(:) + dQ(:))'*XtX*(dP(:) + dQ(:))) <= L*sum(sum((dP.*dP + dQ.*dQ)))
                break;
            else
                L = L*2;
            end
        end
    end
    funcVal = cat(1, funcVal, norm(X*(P(:) + Q(:)) - y)^2 + lambda1*L1infnorm(P) + lambda2*L11norm(Q));
    %fun(iter+1) = norm(X*(P(:) + Q(:)) - y)^2 + lambda1*L1infnorm(P) + lambda2*L11norm(Q);
    
    
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
    
    
    %     % stopping condition
    %     if abs(fun(iter) - fun(iter+1))/fun(iter+1) < tol
    %         count = count + 1;
    %     else
    %         count = 0;
    %     end
    %     if count >= 3
    %         break;
    %     end
    
    % Update the coefficient
    t_new = (1+sqrt(1+4*t_old^2))/2;
    Pn = P + (t_old-1)/t_new*(P - P_old);
    Qn = Q + (t_old-1)/t_new*(Q - Q_old);
    
end

W = P + Q;

end

