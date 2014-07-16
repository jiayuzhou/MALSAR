function mse = eval_MTL_mse (Y, X, W)
%% FUNCTION eval_MTL_mse
%   computation of mean squared error given a specific model.
%   the value is the lower the better.
%   
%% FORMULATION
%   [\sum_i sqrt(sum(X{t} * W(:, t) - Y{t}).^2) *
%   length(Y{t})]/\sum(length(Y{t}))
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
    task_num = length(X);
    mse = 0;
    
    total_sample = 0;
    for t = 1: task_num
        y_pred = X{t} * W(:, t);
        mse = mse + sqrt(sum((y_pred - Y{t}).^2)) * length(y_pred);
        total_sample = total_sample + length(y_pred);
    end
    mse = mse./total_sample;
end
