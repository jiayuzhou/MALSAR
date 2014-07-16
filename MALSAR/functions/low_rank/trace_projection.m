%% FUNCTION trace_projection
%   solves the Trace-norm projection problem.
% 
%% OBJECTIVE
%   argmin_X = 0.5 \|X - L\| + alpha/2 \|L\| 
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
%% RELATED PAPERS
%   [1] Cai et. al. A Singlar Value Thresholding Algorihtm for Matrix
%   Completion.


function [L_hat L_tn] = trace_projection(L, alpha)

[d1 d2] = size(L);

if (d1 > d2)
    
    [U S V] = svd(L, 0);
    
    thresholded_value = diag(S) - alpha / 2;
    
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    
    L_hat = U * diag(diag_S) * V';
    L_tn = sum(diag_S);
else 

    new_L = L';
    
    [U S V] = svd(new_L, 0);
    
    thresholded_value = diag(S) - alpha / 2;
    
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    
    L_hat = U * diag(diag_S) * V';

    L_hat = L_hat';
    L_tn = sum(diag_S);
end
