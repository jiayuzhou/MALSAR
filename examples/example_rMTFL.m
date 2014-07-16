%% file example_rMTFL.m
% this file shows the usage of Least_rMTFL.m function 
% and study how to detect outlier tasks. 
%
%% OBJECTIVE
%  argmin_W ||X(P+Q) - Y||_F^2 + lambda1*||P||_{1,2} + lambda2*||Q^T||_{1,2}
%   s.t. W = P + Q
%
%% Copyright (C) 2012 Jiayu Zhou, and Jieping Ye
%
% You are suggested to first read the Manual.
% For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
% Last modified on April 16, 2012.
%
%% Related papers
%
% [1] Gong, P. and Ye, J. and Zhang, C. Robust Multi-Task Feature Learning,
% Submitted, 2012
%

clear;
clc;
close all;

addpath('../MALSAR/functions/rMTFL/'); % load function 
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
opts.maxIter = 500; % maximum iteration number of optimization.

rho_1 = 90;%   rho1: P
rho_2 = 280; %   rho2: Q

[W funcVal P Q] = Least_rMTFL(X, Y, rho_1, rho_2, opts);



% draw figure
close;
figure();
subplot(3,1,1);
%imshow(1- (abs(S')~=0), 'InitialMagnification', 'fit');
imshow(P'==0, 'InitialMagnification', 'fit')
ylabel('P^T (feature)');
title('Visualization of Robust Multi-Task Feature Learning Model');
subplot(3,1,2);
%imshow(1- (zscore(L')), 'InitialMagnification', 'fit')
imshow(Q'==0, 'InitialMagnification', 'fit')
ylabel('Q^T (outliers)');
subplot(3,1,3);
%imshow(1- (zscore(W')), 'InitialMagnification', 'fit')
imshow(W'==0, 'InitialMagnification', 'fit')
ylabel('W^T');
xlabel('Dimension')
print('-dpdf', '-r600', 'LeastrMTFLExp');
