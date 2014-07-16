%% file example_Trace.m
% this file shows the usage of Least_Trace.m function 
% and study the low-rank patterns. 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2) 
%            + rho1 \|W\|_*}
%  where \|W\|_* = sum(svd(W, 0)) is the trace norm
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
%% Related papers
%
% [1] Ji, S. and Ye, J. An Accelerated Gradient Method for Trace Norm Minimization, ICML 2009
%


clear;
clc;
close;

addpath('../MALSAR/functions/low_rank/'); % load function
addpath('../MALSAR/utils/'); % load utilities

load('../data/school.mat'); % load sample data.

lambda = [1 10 100 200 500 1000 2000];

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.

tn_val = zeros(length(lambda), 1);
rk_val = zeros(length(lambda), 1);
log_lam  = log(lambda);

for i = 1: length(lambda)
    [W funcVal] = Least_Trace(X, Y, lambda(i), opts);
    % set the solution as the next initial point. 
    % this gives better efficiency. 
    opts.init = 1;
    opts.W0 = W;
    tn_val(i) = sum(svd(W));
    rk_val(i) = rank(W);
end

% draw figure
figure;
plot(log_lam, tn_val);
xlabel('log(\rho_1)')
ylabel('Trace Norm of Model (Sum of Singular Values of W)')
title('Trace Norm of Predictive Model when Changing Regularization Parameter');
set(gca,'FontSize',12);
print('-dpdf', '-r100', 'LeastTraceExp');


figure;
plot(log_lam, rk_val);
xlabel('log(\rho_1)')
ylabel('Rank of Model')
title('Rank of Predictive Model when Changing Regularization Parameter');
set(gca,'FontSize',12);
print('-dpdf', '-r100', 'LeastTraceExp_2');