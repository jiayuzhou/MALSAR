%% FUNCTION mtSplitPerc
% Solve the l1 projection problem 
% 
%% OBJECTIVE
%      output = argmin_Z  beta/2gamma ||Z - S||_F^2 + |Z|_1
%
%% INPUT 
%   Sp: S the point to be projected 
%   beta,gamma: projection parameter 
%
%% OUTPUT
%   Tp: projected point 
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jianhui and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%

function Tp = Solve_OneNorm(Sp, beta, gamma)

Tp1 = Sp - 0.5 * gamma / beta; 
Tp1( Tp1 <= 0.5 * gamma / beta ) = 0;

Tp2 = Sp + 0.5 * gamma / beta;
Tp2(Tp2 >= - 0.5 * gamma / beta) = 0;

Tp = Tp1 + Tp2;

