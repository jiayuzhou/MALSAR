%% FUNCTION combine_input
%   transform X, Y cell array input into matrix/vector input with
%   ind and ssize. 
%
%% INPUT
%   X: {[n_i, d]}*t
%   Y: {[n_i, 1]}*t
%
%% OUTPUT
%   Xcmb: [\sum_i n_i, d]
%   Ycmb: [\sum_i n_i, 1]
%   ind: index
%   ssize: the array of size
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

function [Xcmb Ycmb ind ssize] = combine_input(X, Y)

t = length(X);

Xcmb = [];
Ycmb = [];

ind = 1;
ssize = zeros(t, 1);

for i = 1: t
    Xcmb = cat(1, Xcmb,  X{i});
    Ycmb = cat(1, Ycmb,  Y{i});
    ind  = cat(1, ind,   ind(end) + size(X{i},1));
    ssize(i) = size(X{i},1);
end

end