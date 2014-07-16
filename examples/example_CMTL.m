%% file example_CMTL.m
% this file shows the usage of Least_CMTL.m function
% and study how to capture tasks clusters.
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2) 
%            + rho1 * eta (1+eta) trace(W (eta I + M)^-1 W')
%     subject to: trace (M) = k, M \preceq I, M \in S_+^t, eta= rho2/rho1
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
% [1] J. Zhou, J. Chen and J. Ye, Clustered Multi-Task Learning via
% Alternating Structure Optimization, NIPS 2011.
%


clc;
clear;
close all;

addpath('../MALSAR/functions/CMTL/'); % load function
addpath('../MALSAR/utils/'); % load utilities

%rng('default'); % Available from Matlab 2011.

clus_var = 900;  % cluster variance
task_var = 16;   % inter task variance
nois_var = 150;  % variance of noise

clus_num = 2;                        % clusters
clus_task_num = 10;                  % task number of each cluster
task_num = clus_num * clus_task_num; % total task number.
sample_size = 100;
dimension   = 20;        % total dimension
comm_dim    = 2;         % independent dimension for all tasks.
clus_dim    = floor((dimension - comm_dim)/2); % dimension of cluster

% generate cluster model
cluster_weight = randn(dimension, clus_num) * clus_var;
for i = 1: clus_num
    cluster_weight (randperm(dimension-clus_num)<=clus_dim, i) = 0;
end
cluster_weight (end-comm_dim:end, :) = 0;
W = repmat (cluster_weight, 1, clus_task_num);
cluster_index = repmat (1:clus_num, 1, clus_task_num)';

% generate task and intra-cluster variance
W_it = randn(dimension, task_num) * task_var;
for i = 1: task_num
    W_it(cat(1, W(1:end-comm_dim, i)==0, zeros(comm_dim, 1))==1, i) = 0;
end
W = W + W_it;

% apply noise;
W = W + randn(dimension, task_num) * nois_var;

%%%% Generate Input/Output
X = cell(task_num, 1);
Y = cell(task_num, 1);
for i = 1: task_num
    X{i} = randn(sample_size, dimension);
    xw   = X{i} * W(:, i);
    xw   = xw + randn(size(xw)) * nois_var;
    Y{i} = sign(xw);
end

opts.init = 0;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-6;   % tolerance.
opts.maxIter = 1500; % maximum iteration number of optimization.

rho_1 = 10;
rho_2 = 10^-1;

W_learn = Least_CMTL(X, Y, rho_1, rho_2, clus_num, opts);

kmCMTL_OrderedModel = zeros(size(W));
OrderedTrueModel = zeros(size(W));


for i = 1: clus_num
    clusModel = W_learn        (:, i:clus_num:task_num );
    kmCMTL_OrderedModel        (:, (i-1)* clus_task_num + 1: i* clus_task_num ) = clusModel;
    
    clusModel = W              (:, i:clus_num:task_num );
    OrderedTrueModel           (:, (i-1)* clus_task_num + 1: i* clus_task_num ) = clusModel;
end

figure;
imshow(1-corrcoef(OrderedTrueModel), 'InitialMagnification', 'fit')
%print -painters -dpdf -r1000 Figure/correlation_groundTruth.pdf
title('Model Correlation: Ground Truth')

figure;
imshow(1-corrcoef(kmCMTL_OrderedModel), 'InitialMagnification', 'fit')
%print -painters -dpdf -r1000 Figure/correlation_kmCMTL.pdf
title('Model Correlation: Clustered MTL');

