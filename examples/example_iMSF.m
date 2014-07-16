%% file example_iMSF.m
%   a toy example on using the iMSF method for learning from block-wise
%   missing data using Logistic_iMSF.m function 
%
%% OBJECTIVE
%   see manual
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
%   Copyright (C) 2011 - 2012 Lei Yuan, Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% RELATED PAPERS
%   [1] Lei Yuan, Yalin Wang, Paul M. Thompson, Vaibhav A. Narayan and Jieping
%       Ye, Multi-Source Learning for Joint Analysis of Incomplete
%       Multi-Modality Neuroimaging Data, KDD 2012
%   [2] Lei Yuan, Yalin Wang, Paul M. Thompson, Vaibhav A. Narayan and Jieping
%       Ye, for the Alzheimer's Disease Neuroimaging Initiative, Multi-source
%       Feature Learning for Joint Analysis of Incomplete Multiple Heterogeneous
%       Neuroimaging Data, NeuroImage 2012 Jul 2; 61(3):622-632.
%

clear, clc;

addpath('../MALSAR/utils/')
addpath('../MALSAR/functions/iMSF')

% construct data sources that are block-wise missing.
n = 50;
p1 = 45;
p2 = 55;
p3 = 50;

X1 = randn(n, p1);
X2 = randn(n, p2);
X3 = randn(n, p3);
X1(round(0.8 * n):end, :) = nan;
X2(1:round(0.2 * n), :) = nan;
X3(round(0.5 * n):round(0.8 * n), :) = nan;
Y = sign(randn(n, 1));
% generate cell input. 
X_Set{1} = X1;
X_Set{2} = X2;
X_Set{3} = X3;

opts.tol = 1e-6;
opts.maxIter = 5000;
lambda = 0.1;

[logistic_Sol, logistic_funVal] = Logistic_iMSF(X_Set, Y, lambda, opts);
[least_Sol,    least_funVal] = Least_iMSF(X_Set, Y, lambda, opts);

