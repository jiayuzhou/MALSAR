%% file example_L21.m
% this file shows the usage of Least_L21.m function 
% and study the group sparsity patterns. 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2) 
%            + rho1 * \|W\|_{2,1} + opts.rho_L2 * \|W\|_F^2}
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
% [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
% [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
% Report, 2010.
%

clear;
clc;
close;

addpath('../MALSAR/functions/joint_feature_learning/'); % load function
addpath('../MALSAR/utils/'); % load utilities

load('../data/school.mat'); % load sample data.
d = size(X{1}, 2);  % dimensionality.

lambda = [200 :300: 1500];

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.

sparsity = zeros(length(lambda), 1);
log_lam  = log(lambda);

for i = 1: length(lambda)
    [W funcVal] = Least_L21(X, Y, lambda(i), opts);
    % set the solution as the next initial point. 
    % this gives better efficiency. 
    opts.init = 1;
    opts.W0 = W;
    sparsity(i) = nnz(sum(W,2 )==0)/d;
end

% draw figure
h = figure;
plot(log_lam, sparsity);
xlabel('log(\rho_1)')
ylabel('Row Sparsity of Model (Percentage of All-Zero Columns)')
title('Row Sparsity of Predictive Model when Changing Regularization Parameter');
set(gca,'FontSize',12);
print('-dpdf', '-r100', 'LeastL21Exp');
