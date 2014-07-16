%% file example_Dirty.m
% this file shows the usage of Least_Dirty.m function 
% and study the group sparsity patterns. 
%
%% OBJECTIVE
% argmin_W ||X(P+Q) - Y||_F^2 + lambda1*||P||_{1,inf} + lambda2*||Q||_{1,1}
%  s.t. W = P + Q
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
% [1] Jalali, A. and Ravikumar, P. and Sanghavi, S. and Ruan, C. A dirty
% model for multi-task learning, NIPS 2010.
%

clear;
clc;
close;

addpath('../MALSAR/functions/dirty/'); % load function
addpath('../MALSAR/c_files/prf_lbm/'); % load projection c libraries. 
addpath('../MALSAR/utils/'); % load utilities

%rng('default');     % reset random generator. Available from Matlab 2011.

%generate synthetic data.
dimension = 500;
sample_size = 50;
task = 50;
X = cell(task ,1);
Y = cell(task ,1);
for i = 1: task
    X{i} = rand(sample_size, dimension);
    Y{i} = rand(sample_size, 1);
end

opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-4;   % tolerance. 
opts.maxIter = 500; % maximum iteration number of optimization.

rho_1 = 350;%   rho1: group sparsity regularization parameter
rho_2 = 10;%   rho2: elementwise sparsity regularization parameter

[W funcVal P Q] = Least_Dirty(X, Y, rho_1, rho_2, opts);



% draw figure
close;
figure();
subplot(3,1,1);
imshow(1- (abs(P')~=0), 'InitialMagnification', 'fit');
ylabel('P^T');
title('Visualization Non-Zero Entries in Dirty Model');
subplot(3,1,2);
imshow(1- (abs(Q')~=0), 'InitialMagnification', 'fit')
ylabel('Q^T');
subplot(3,1,3);
imshow(1- (abs(W')~=0), 'InitialMagnification', 'fit')
ylabel('W^T');
xlabel('Dimension')

