%% FUNCTION line_search_cond_RMTL
%   line search condition of robust multi-task learning formulation.
%
%% INPUT
%   L_hat, S_hat: the pair of solution points
%   P, Q: the pair of searching points
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

function [sign left] = line_search_cond_RMTL( L_hat, S_hat, P, Q, X, Y, derivative, gamma, opts)

m = size(L_hat, 2);

f_T = 0 ; f_S = 0;

if opts.pFlag
    parfor ii = 1:m
        Xi = X{ii};
        f_T = f_T + norm( Xi' * (L_hat(:, ii) + S_hat(:, ii)) - Y{ii})^2;
        f_S = f_S + norm( Xi' * (     P(:, ii) +     Q(:, ii)) - Y{ii})^2;
        %f_T = norm( X' * (L_hat + S_hat) - Y, 'fro' )^2;
        %f_S = norm( X' * (P + Q) - Y, 'fro' )^2;
    end
else
    for ii = 1:m
        Xi = X{ii};
        f_T = f_T + norm( Xi' * (L_hat(:, ii) + S_hat(:, ii)) - Y{ii})^2;
        f_S = f_S + norm( Xi' * (     P(:, ii) +     Q(:, ii)) - Y{ii})^2;
        %f_T = norm( X' * (L_hat + S_hat) - Y, 'fro' )^2;
        %f_S = norm( X' * (P + Q) - Y, 'fro' )^2;
    end
end

fro_term = gamma / 2 * ( norm( L_hat - P, 'fro' )^2 + norm( S_hat - Q, 'fro' )^2 );

left = f_T;
%right = f_S + trace( (L_hat - P)' * derivative ) + trace( (S_hat - Q)' * derivative ) + fro_term;
LhP = L_hat - P;
ShQ = S_hat - Q;
right = f_S + sum(sum( LhP .* derivative )) ...
    + sum(sum( ShQ .* derivative )) + fro_term;

if (left <= right)
    sign = 0;   % line search is done
else
    sign = 1;    % contine to do line search
end