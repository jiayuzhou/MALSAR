%% FUNCTION Least_iMSF
%   Incompletet Multi-Source Learning with Least Squares Loss (interface). This
%   code generates structural information for MultiSource_LeastR and 
%   performs the optimization. 
%
%% OBJECTIVE
%   see manual.
%
%% INPUT
%   A_Set: {n * d} * t - input. The cell array of data. Each cell should be 
%          n by p_i, with missing samples denoted by a whole row of NaNs.
%   Y_set: {n * 1} * t - output. Y_i \in {-1, 1}.
%   z:     group sparsity regularization parameter (a relative value, [0,1])
%
%% OUTPUT
%   Sol: {struct} solution struct array.
%         For task i: model: struct{i}.x, bais: struct{i}.c
%   funVal: objective function values at each iteration.
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
%   Copyright (C) 2011 - 2012 Lei Yuan, Jiayu Zhou, and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 12, 2012.
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
%% RELATED FUNCTIONS
%   init_opts, Construct_iMSF, MultiSource_LeastR

function [Sol, funVal] = Least_iMSF(X_Set, Y, lambda, opts) 

if nargin< 4
    opts = [];
end

[A_Set, Y_Set, W, G, ind] = Construct_iMSF(X_Set, Y);
[Sol, funVal] = MultiSource_LeastR(A_Set, Y_Set, lambda, G, W, ind, opts);