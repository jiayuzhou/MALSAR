%% FUNCTION example_LSSMTC.m
%   Example of multi-task clustering. 
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
%   Copyright (C) 2011 - 2012 Quanquan Gu, Jiayu Zhou, and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 17, 2012.
%
%% RELATED PAPERS
%   [1]Quanquan Gu, Jie Zhou. Learning the shared subspace
%      for multi-task clustering and transductive transfer
%      classification. In Proc. of the International Conference
%      on Data Mining (ICDM), Miami, Florida, USA, 2009.

clear
close all
clc

addpath('../MALSAR/functions/mutli-task clustering/');
addpath('../MALSAR/utils/');

%===========================================
%This is a demo on comp v.s. sci data set

path = '../data/Newsgroup/comp.vs.sci/';

m = 2;
cellX = cell(m,1);
cellgnd = cell(m,1);
for i=1:m
    task_data = load([path 'Task' num2str(i)]);
    cellX{i} = task_data.fea';
    % if the evaluation metrices are not needed 
    % then cellgnd can be set to []. 
    cellgnd{i} = task_data.gnd;
end

c = length(unique(cellgnd{1})); % cluster number 

opts.tFlag = 2; % termination: run maximum iteration. 
opts.maxIter = 20; % maximum iteration.


l = 2; % subspace dimension 
lambda = 0.75; % multi-task regulariation paramter 
[W cellM cellP residue cellAcc cellNMI] = LSSMTC(cellX,cellgnd,c,l,lambda,opts);


figure
for i = 1:m
    subplot(1,m,i)
    plot(cellAcc{i})
    ylabel('accuracy')
    xlabel('iteration')
    title(sprintf('Task %u', i))
end
print('-dpdf', '-r600', 'LSSMTC_acc');

figure
for i = 1:m
    subplot(1,m,i)
    plot(cellNMI{i})
    ylabel('normalized mutual information')
    xlabel('iteration')
    title(sprintf('Task %u', i))
end
print('-dpdf', '-r600', 'LSSMTC_nmi'); 




