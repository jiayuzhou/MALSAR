%% FUNCTION nchoose
%   How many unique combinations of 1 to N elements are there. This 
%   function is not intended to directly called by users. 
%   See the Logistic_iMSF function.
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
%% RELATED FUNCTIONS
%   init_opts, Construct_iMSF, Logistic_iMSF, MultiSource_LogisticR

function W = nchoose(S)

N = numel(S) ; 

% How many unique combinations of 1 to N elements are there
M = (2^N)-1 ;
if N > 18,
    warning('Nchoose:LargeOutput', ...
        'There are %d unique combinations. Please be patient ...',M) ;
end

S = S(:).' ;   % make the set a row vector, for uniform output

W = cell(M,1) ;    % Pre-allocation of output
p2=2.^(N-1:-1:0) ; % This part of the formula can be taken out of the loop

for i=1:M,
    % calculate the (reversed) binary representation of i
    % select the elements of the set based on this representation
    W{i} = S(bitget(i*p2,N) > 0) ; 
end