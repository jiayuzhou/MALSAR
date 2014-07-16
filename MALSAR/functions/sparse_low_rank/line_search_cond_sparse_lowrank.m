%% FUNCTION line_search_cond_sparse_lowrank
%   line search condition of sparse + low rank learning formulation.
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jianhui Chen and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
function [sign qp_term] = line_search_cond_sparse_lowrank(Tp, Tq, Sp, Sq, X, Y, derivative, L, opts)

m = size(Tp, 2);

f_T = 0 ; f_S = 0;

if opts.pFlag
    parfor ii = 1:m
        Xi = X{ii};
        f_T = f_T + norm( Xi' * (Tp(:, ii) + Tq(:, ii)) - Y{ii})^2;
        f_S = f_S + norm( Xi' * (Sp(:, ii) + Sq(:, ii)) - Y{ii})^2;
        
    end
else
    for ii = 1:m
        Xi = X{ii};
        f_T = f_T + norm( Xi' * (Tp(:, ii) + Tq(:, ii)) - Y{ii})^2;
        f_S = f_S + norm( Xi' * (Sp(:, ii) + Sq(:, ii)) - Y{ii})^2;
    end
end

fro_term = L / 2 * ( norm( Tp - Sp, 'fro' )^2 + norm( Tq - Sq, 'fro' )^2 );

left = f_T;

%right = f_S + trace( (Tp - Sp)' * derivative ) + trace( (Tq - Sq)' *
%derivative ) + fro_term;


TSp = Tp - Sp;
TSq = Tq - Sq;
right = f_S + sum(sum(TSp .* derivative)) + sum(sum(TSq .* derivative)) + fro_term;


qp_term = left;

if (left <= right)
    sign = 0;   % line search is done
else
    sign = 1;    % contine to do line search
end