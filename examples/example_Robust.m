%% file example_Robust.m
% this file shows the usage of Least_RMTL.m function 
% and study how to detect outlier tasks. 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|L\|_* + rho2 * \|S\|_{1, 2}}
% where W = L + S
%       \|S\|_{1, 2} = sum( sum(S.^2) .^ 0.5 )
%       \|L\|_*      = sum( svd(L, 0) )
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
% [1] Chen, J., Zhou, J. and Ye, J. Integrating Low-Rank and Group-Sparse
% Structures for Robust Multi-Task Learning, KDD 2011
%

clear;
clc;
close;

addpath('../MALSAR/functions/robust/'); % load function 
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
opts.tol = 10^-6;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.

rho_1 = 10;%   rho1: low rank component L trace-norm regularization parameter
rho_2 = 30; %   rho2: sparse component S L1,2-norm sprasity controlling parameter

[W funcVal L S] = Least_RMTL(X, Y, rho_1, rho_2, opts);



% draw figure
close;
figure();
subplot(3,1,1);
%imshow(1- (abs(S')~=0), 'InitialMagnification', 'fit');
imshow((abs(S')/max(max(abs(S')))), 'InitialMagnification', 'fit')
ylabel('S^T (outliers)');
title('Visualization of Robust Multi-Task Learning Model');
subplot(3,1,2);
%imshow(1- (zscore(L')), 'InitialMagnification', 'fit')
imshow((abs(L')/max(max(abs(L')))), 'InitialMagnification', 'fit')
ylabel('L^T (low rank)');
subplot(3,1,3);
%imshow(1- (zscore(W')), 'InitialMagnification', 'fit')
imshow((abs(W')/max(max(abs(W')))), 'InitialMagnification', 'fit')
ylabel('W^T');
xlabel('Dimension')
colormap(Jet)
print('-dpdf', '-r600', 'LeastRMTLExp');
