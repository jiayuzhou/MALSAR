%% FUNCTION Least_RMTL
%   Robust Multi-Task Learning with Least Squares Loss.
%
%% OBJECTIVE
%   argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|L\|_* + rho2 * \|S\|_{1, 2}}
%   where W = L + S
%       \|S\|_{1, 2} = sum( sum(S.^2) .^ 0.5 )
%       \|L\|_*      = sum( svd(L, 0) )
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: low rank component L trace-norm regularization parameter
%   rho2: sparse component S L1,2-norm sprasity controlling parameter
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
%   L_hat: low rank component (task relatedness)
%   S_hat: group sparse component (outlier detection)
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
%% RELATED PAPERS
%
%   [1] Chen, J., Zhou, J. and Ye, J. Integrating Low-Rank and Group-Sparse
%   Structures for Robust Multi-Task Learning, KDD 2011
%
%% RELATED FUNCTIONS
%   init_opts

function [W funcVal L_hat S_hat L0 S0] = Least_RMTL(X, Y, rho1, rho2, opts)

if nargin <4
    error('\n Inputs: X, Y, rho1, and rho2 should be specified!\n');
end
X = multi_transpose(X);

if nargin <5
    opts = [];
end

% initialize options.
opts=init_opts(opts);


d = size(X{1}, 1); m = length(Y);

% Initialize L and S
if opts.init==2
    L0 = zeros(d, m);
    S0 = zeros(d, m);
elseif opts.init == 0
    L0 = randn(d, m);
    S0 = randn(d, m);
else
    if isfield(opts,'L0')
        L0=opts.L0;
        if (nnz(size(L0)-[d, m]))
            error('\n Check the input .L0');
        end
    else
        L0=randn(d, m);
    end
    
    if isfield(opts,'S0')
        S0=opts.S0;
        if (nnz(size(S0)-[d, m]))
            error('\n Check the input .S0');
        end
    else
        S0=randn(d, m);
    end
    
end

L_i = L0;
S_i = L0;
L_i_min_1 = L0;
S_i_min_1 = S0;

% Set an array to save the objective value
funcVal = [ ];

t_i_min_2 = 0;
t_i_min_1 = 1;

% Used for computing gradient
XY = cell(m, 1);
for jj = 1:m
    XY{jj} = X{jj} * Y{jj};
end

gamma = 1;
gamma_inc = 2;
% main loop
% ++++++++++++++++++++++++++++++++++++++++++
for iter = 1 : opts.maxIter
    
    alpha_i = ( t_i_min_2 - 1 ) / t_i_min_1;
    
    P  = ( 1 + alpha_i ) * L_i - alpha_i * L_i_min_1;
    Q = ( 1 + alpha_i ) * S_i - alpha_i * S_i_min_1;
    
    derivative = zeros(d, m);
    if opts.pFlag
        parfor kk = 1:m
            Xi = X{kk};
            col = P(:, kk) + Q(:, kk);
            derivative(:, kk) = 2 * Xi * (Xi' * col)  - 2 * XY{kk};
        end
    else
        for kk = 1:m
            Xi = X{kk};
            col = P(:, kk) + Q(:, kk);
            derivative(:, kk) = 2 * Xi * (Xi' * col)  - 2 * XY{kk};
        end
    end
    
    sign = 1;
    
    while (sign)
        
        new_L = P - derivative / gamma;
        new_S = Q - derivative / gamma;
        
        % Compute project-L
        L_hat = solve_trace_norm_RMTL(new_L, 2*rho1/gamma);
        
        % Compute project-S
        S_hat = solve_12_norm(new_S, 2*rho2/gamma, d, m);
        
        [sign qp_term] = line_search_cond_RMTL(L_hat, S_hat, P, Q, X, Y, derivative, gamma, opts);
        
        if (sign)
            gamma = gamma * gamma_inc;
        end
        
    end
    
    tmp0 = qp_term; % norm( X' * (L_hat + S_hat) - Y, 'fro' )^2;
    tmp1 = rho1 * sum( svd(L_hat, 0) );
    tmp2 = rho2 * sum( sum(S_hat.^2) .^ 0.5 );
    
    funcVal = cat(1, funcVal,  tmp0 + tmp1 + tmp2 );
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    
    t_i_min_2 = t_i_min_1;
    t_i_min_1 = 0.5 * ( 1 + ( 1+4 * t_i_min_1^2 )^0.5 ) ;
    
    L_i_min_1 = L_i;
    L_i = L_hat;
    
    S_i_min_1 = S_i;
    S_i = S_hat;
    
end % for loop

W = L_hat + S_hat;

end

