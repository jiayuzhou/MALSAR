%% file example_SRMTL.m
% this file shows the usage of Least_SRMTL.m function 
% and study the graph regularization 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2) 
%            + rho1 * norm(W*R, 'fro')^2 + rho2 * \|W\|_1}
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
% [1] Zhou, J. Technical Report. http://www.public.asu.edu/~jzhou29/Software/SRMTL/CrisisEventProjectReport.pdf
%

clear;
clc;
close;

addpath('../MALSAR/functions/Lasso/'); % load function
addpath('../MALSAR/functions/SRMTL/'); % load function
addpath('../MALSAR/utils/'); % load utilities

%rng('default');     % reset random generator.Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 500; % maximum iteration number of optimization.

load('../data/school.mat'); % load sample data.
task_num = length(X);
% use Lasso calculate a model (used for graph analysis)
[W_pre] = Least_Lasso(X, Y, 0.01, opts);
% normalize matrix.
mean_1=mean(W_pre,1);
W_pre=W_pre-repmat(mean_1,size(W_pre, 1),1);
norm_2=sqrt( sum(W_pre.^2,1) );
W_pre=W_pre./repmat(norm_2,size(W_pre, 1),1);
% use model correlation to calculate a graph
correlation_threshold = 0.85;
graph = corrcoef(W_pre)>correlation_threshold;
graph = graph - eye(task_num);
edge_num = nnz(graph)/2;
fprintf('%u edges are found\n', edge_num);

imshow(1- graph, 'InitialMagnification', 'fit')
title(sprintf('Correlation Graph with Threshold %.2f (#edge = %u)', correlation_threshold, edge_num));
print('-dpdf', '-r300', 'LeastSRMTLExp_1');

imshow(1- corrcoef(W_pre), 'InitialMagnification', 'fit')
title('Pairwise Correlation for School Data');
colormap(autumn)
print('-dpdf', '-r300', 'LeastSRMTLExp_2');

% construct graph structure variable.
R = [];
for i = 1: task_num
    for j = i + 1: task_num
        if graph (i, j) ~=0
            edge = zeros(task_num, 1);
            edge(i) = 1;
            edge(j) = -1;
            R = cat(2, R, edge);
        end
    end
end

%%% There can be other choices of the R (for different structures).
%R = eye(task_num); % ridge penalty.
%R = zeros(t,t-1);H(1:(t+1):end)=1;H(2:(t+1):end)=-1; % order structure
%R = eye (task_num) - ones (task_num) / task_num;  %regularized MTL penalty

[W_est funcVal] = Least_SRMTL(X, Y, R, 1, 20);

%plot(funcVal)



