%% file LeastmsmtflExp.m
% this file shows the usage of Least_msmtfl_capL1.m function 
% and study how to detect outlier tasks. 
%
%% OBJECTIVE
%   min_W ||XW - Y||_F^2 + lambda*\sum_i \min{||W^i||_1,theta}
%
%   It's a nonconvex optimization problem, which can be relaxed into a Multi-Stage Convex optimization:
%   min_W ||XW - Y||_F^2 + \sum_i{gamma_i*||W^i||_1}
%
%% Copyright (C) 2012 Jiayu Zhou, Pinghua Gong, and Jieping Ye
%
% You are suggested to first read the Manual.
% For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
% Last modified on Dec 18, 2012.
%
%% Related papers
%
% [1] Pinghua Gong, Jieping Ye, Changshui Zhang. Multi-Stage Multi-Task 
%     Feature Learning. The 26th Annual Conference on Neural Information 
%     Processing Systems (NIPS 2012), Lake Tahoe, Nevada, USA, 
%     December 3-6, 2012.
%


clear
clc;
close all;

addpath('../MALSAR/functions/msmtfl/'); % load function 
addpath('../MALSAR/utils/'); % load utilities

%rng('default');     % reset random generator. Available from Matlab 2011.

m = 20; % task number
n = 30; % number of samples for each task
samplesize = n*ones(m,1);
d = 200; % dimensionality

zerorow_percent = 0.9;
restzero_percent = 0.8;
noiselevel = 0.001;
%totalnum = sum(samplesize);

opts.tol = 1e-5;
opts.lFlag = 0;
opts.init = 1;
opts.W0 = randn(d,m);
opts.tFlag = 1;

scale = 50;

para = [0.00005; 0.0001; 0.0002; 0.0005]*sqrt(log(d*m)/n);
para_num = length(para);
repeat_num = 10;
maxstep = 10;
Werror_ms1_21 = zeros(maxstep,repeat_num); Werror_ms2_21 = Werror_ms1_21; Werror_ms3_21 = Werror_ms1_21; Werror_ms4_21 = Werror_ms1_21;

for jj = 1:repeat_num
    % generate model
    W = rand(d,m)*20 - 10;
    permnum = randperm(d);
    zerorow = permnum(1:round(d*zerorow_percent));
    nonzerorow = permnum(round(d*zerorow_percent)+1:end);
    W(zerorow,:) = 0;
    Wtemp = W(nonzerorow,:);
    permnum = randperm(length(nonzerorow)*m);
    Wtemp(permnum(1:round(length(nonzerorow)*m*restzero_percent))) = 0;   
    W(nonzerorow,:) = Wtemp;
    
    % genetate data    
    X = cell(m,1);
    Y = cell(m,1);
    for ii = 1:m
        X{ii} = normrnd(0,1,samplesize(ii),d);
        X{ii} = normalize(X{ii},samplesize(ii));
        Y{ii} = X{ii} * W(:, ii) + noiselevel*normrnd(0,1,samplesize(ii),1);
    end
    
    for ii = 1:maxstep
        fprintf('[Rep: %u] Max iteration: %u\n', jj, ii);
        opts.maxIter = ii;
        
        lambda = para(1)*m*n;
        theta = scale(1)*m*lambda;
        [W_ms1,~] = Least_msmtfl_capL1(X,Y,lambda,theta,opts);
        Werror_ms1_21(ii,jj) = norm(sum(abs(W_ms1 - W),2));
        
        lambda = para(2)*m*n;
        theta = scale(1)*m*lambda;
        [W_ms2,~] = Least_msmtfl_capL1(X,Y,lambda,theta,opts);
        Werror_ms2_21(ii,jj) = norm(sum(abs(W_ms2 - W),2));

        lambda = para(3)*m*n;
        theta = scale(1)*m*lambda;   
        [W_ms3,~] = Least_msmtfl_capL1(X,Y,lambda,theta,opts);
        Werror_ms3_21(ii,jj) = norm(sum(abs(W_ms3 - W),2));

        lambda = para(4)*m*n;
        theta = scale(1)*m*lambda;   
        [W_ms4,~] = Least_msmtfl_capL1(X,Y,lambda,theta,opts);
        Werror_ms4_21(ii,jj) = norm(sum(abs(W_ms4 - W),2));
    end
end

figure;
plot(1:maxstep,mean(Werror_ms1_21,2),'ro--',...
     1:maxstep,mean(Werror_ms2_21,2),'gs:',...
     1:maxstep,mean(Werror_ms3_21,2),'bd-',...
     1:maxstep,mean(Werror_ms4_21,2),'cv-.')
xlabel('Stage')
ylabel('Parameter estimation error (L2,1)')
legend(['\alpha=',num2str(para(1))],['\alpha=',num2str(para(2))],['\alpha=',num2str(para(3))],['\alpha=',num2str(para(4))])
title(['m=',num2str(m),',n=',num2str(n),',d=',num2str(d),',\sigma=',num2str(noiselevel)])

set(findobj('Type','line'),'LineWidth',1)
 
print('-dpdf', '-r600', 'LeastmsmtflExp');



