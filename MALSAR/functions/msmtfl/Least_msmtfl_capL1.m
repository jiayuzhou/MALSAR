%% FUNCTION Least_msmtfl_capL1
%   Multi-Stage Multi-Task Feature Learning
%
%% OBJECTIVE
%   min_W ||XW - Y||_F^2 + lambda*\sum_i \min{||W^i||_1,theta}
%
%   It's a nonconvex optimization problem, which can be relaxed into a Multi-Stage Convex optimization:
%   min_W ||XW - Y||_F^2 + \sum_i{gamma_i*||W^i||_1}
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   lambda: regularized parameter (vector)
%   theta: theresholding paramter
%
%   (Optional)
%   opts.lFlag: estimate the upper bound of Lipschitz constant if nonzero, zero otherwise 
%
%% OUTPUT
%   W: output weight
%   fun: function values
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as pubtolwlished by
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
%   Last modified on Dec 18, 2012.
%
%
%% RELATED PAPERS
%
% [1] Pinghua Gong, Jieping Ye, Changshui Zhang. Multi-Stage Multi-Task 
%     Feature Learning. The 26th Annual Conference on Neural Information 
%     Processing Systems (NIPS 2012), Lake Tahoe, Nevada, USA, 
%     December 3-6, 2012.
%
%% RELATED FUNCTIONS
%  init_opts, combine_input (utils)

function [W,funcVal] = Least_msmtfl_capL1(X, Y, lambda, theta, opts)

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
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = randn(dimension, task_num);
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .P0');
        end
    else
        W0=randn(dimension, task_num);
    end
end

% Set an array to save the objective value
funcVal = [];

W = W0;

[d,m] = size(W); % d: dimension, m: the number of tasks
X = diagonalize(X,samplesize);
XtX = X'*X; Xty = X'*y;

L1norm = max(sum(abs(X),1)); Linfnorm = max(sum(abs(X),2));

if lFlag
    % Upper bound for largest eigenvalue of Hessian matrix
    L = 2*min([L1norm*Linfnorm; size(X,1)*Linfnorm*Linfnorm; size(X,2)*L1norm*L1norm; size(X,1)*size(X,2)*max(abs(X(:)))]);
else
    % Lower bound for largest eigenvalue of Hessian matrix
    L = 2*max(L1norm*L1norm/size(X,1),Linfnorm*Linfnorm/size(X,2));
end
% Initial function value
funcVal = cat(1, funcVal, norm(X*W(:) - y)^2 + lambda*(sum(min(sum(abs(W),2),theta))));

tolw = 1e-5; % precision for inner iterations. 

for iter = 1:opts.maxIter
    if iter == 1
        weight = lambda*ones(d,m);
    else
        weight = lambda*(repmat(sum(abs(W),2) < theta,1,m));
    end
    [W,~,iterw] = wLassomtl(X,y,XtX,Xty,weight,W,tolw,100,L,lFlag);
    

    if iterw == 1
        tolw = tolw/4;
    end
    
    funcVal = cat(1, funcVal, norm(X*W(:) - y)^2 + lambda*(sum(min(sum(abs(W),2),theta))));
    
    % stopping condition
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
    
end


