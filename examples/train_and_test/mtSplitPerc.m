%% FUNCTION mtSplitPerc
%   Multi-task data set splitting by percentage. 
%   
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   percent: percentage of the splitting range (0, 1)
%
%% OUTPUT
%   X_sel: the split of X that has the specifid percent of samples 
%   Y_sel: the split of Y that has the specifid percent of samples 
%   X_res: the split of X that has the 1-percent of samples 
%   Y_res: the split of Y that has the 1-percent of samples 
%   selIdx: the selection index of for X_sel and Y_sel for each task
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

function [X_sel, Y_sel, X_res, Y_res, selIdx] = mtSplitPerc(X, Y, percent)

if percent > 1 || percent <0
    error('splitting percentage error')
end

task_num = length(X);

selIdx = cell(task_num, 0);
X_sel = cell(task_num, 0);
Y_sel = cell(task_num, 0);
X_res = cell(task_num, 0);
Y_res = cell(task_num, 0);

for t = 1: task_num
    task_sample_size = length(Y{t});
    tSelIdx = randperm(task_sample_size) <task_sample_size * percent;
    
    selIdx{t} = tSelIdx;
    
    X_sel{t} = X{t}(tSelIdx,:);
    Y_sel{t} = Y{t}(tSelIdx,:);
    X_res{t} = X{t}(~tSelIdx,:);
    Y_res{t} = Y{t}(~tSelIdx,:);
    
end