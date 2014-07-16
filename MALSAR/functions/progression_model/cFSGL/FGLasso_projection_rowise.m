%% FUNCTION FGLasso_projection_rowise
%   projection of sparse fused group Lasso.
%
%% OBJECTIVE
%   argmin_w { 0.5 \|w - v\|_2^2
%          + lambda_1 * \|w\|_1 + lambda_2 * \|w*R\|_1  +
%          + lambda_3 * \|w\|_2 }
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jun Liu and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%


function w = FGLasso_projection_rowise(v, lambda_1, lambda_2, lambda_3)

% starting point (dual variable).
w0 = zeros(length(v)-1, 1);

%% 1st Projection:
%    w_1 = argmin_w { 0.5 \|w - v\|_2^2
%          + lambda_1 * \|w\|_1 + lambda_2 * \|w*R\|_1}
% This is equivalent to solving a fused Lasso problem. 
w_1 = flsa(v, w0,  lambda_1, lambda_2, length(v), 1000, 1e-9, 1, 6);

%% 2nd Projection:
% argmin_w { 0.5 \|w - w_1\|_2^2
%          + lambda_3 * \|w_1\|_2 }
% This is a simple thresholding:
%    w_2 = max(\|w_1\|_2 - \lambda_3, 0)/\|w_1\|_2 * w_1
nm = norm(w_1, 2);
if nm == 0
    w_2 = zeros(size(w_1));
else
    w_2 = max(nm - lambda_3, 0)/nm * w_1;
end

w = w_2;

end