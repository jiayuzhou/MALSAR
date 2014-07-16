%% FUNCTION Least_SparseTrace
%   Incoherent Sparse and Low-Rank Learning with Least Squares Loss.
%
%% OBJECTIVE
%   argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + gamma * \|P\|_1 }
%   subject to: W = P + Q, \|Q\|_* <= tau
%   where \|Q\|_*      = sum( svd(Q, 0) )
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   gamma: sparse component P L1,2-norm sprasity controlling parameter
%   tau: low rank component Q trace-norm regularization parameter
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
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
%% RELATED PAPERS
%
%   [1] Chen, J., Liu, J. and Ye, J. Learning Incoherent Sparse and Low-Rank
%   Patterns from Multiple Tasks, KDD 2010
%
%% RELATED FUNCTIONS
%   init_opts


function [W funcVal Tp_val Tq_val] = Least_SparseTrace(X, Y, gamma, tau, opts )


if nargin <4
    error('\n Inputs: X, Y, rho1, and rho2 should be specified!\n');
end
X = multi_transpose(X);

if nargin <5
    opts = [];
end

% initialize options.
opts=init_opts(opts);


zero_threshold = 0;

d = size(X{1}, 1); m = length(Y);


% Initialize L and S
if opts.init==2
    P0 = zeros(d, m);
    Q0 = zeros(d, m);
elseif opts.init == 0
    P0 = randn(d, m);
    Q0 = P0;
else
    if isfield(opts,'P0')
        P0=opts.P0;
        if (nnz(size(P0)-[d, m]))
            error('\n Check the input .L0');
        end
    else
        P0=randn(d, m);
    end
    
    if isfield(opts,'Q0')
        Q0=opts.Q0;
        if (nnz(size(Q0)-[d, m]))
            error('\n Check the input .S0');
        end
    else
        Q0=P0;
    end
    
end


% Initialize
Tp_i = P0;
Tq_i = Q0;
Tp_i_min_1 = P0;
Tq_i_min_1 = Q0;

% initial parameters setting
%  ++++++++++++++++++++++++++++++++++++++++++

% Initialize L
Li = 1;



% Set an array to save the objective value
funcVal = [ ];

% Set auxiliary parameters

t_i_min_2 = 0;
t_i_min_1 = 1;
XY = cell(m, 1);
for jj = 1:m
    XY{jj} = X{jj} * Y{jj};
end

% main loop
% ++++++++++++++++++++++++++++++++++++++++++
for iter = 1 : opts.maxIter
    
    alpha_i = ( t_i_min_2 - 1 ) / t_i_min_1;
    
    Sp = ( 1 + alpha_i ) * Tp_i - alpha_i * Tp_i_min_1;
    Sq = ( 1 + alpha_i ) * Tq_i - alpha_i * Tq_i_min_1;
    
    derivative = zeros(d, m);
    if opts.pFlag
        parfor kk = 1:m
            Xi = X{kk};
            col = Sp(:, kk) + Sq(:, kk);
            derivative(:, kk) = 2 * Xi * (Xi' * col)  - 2 * XY{kk};
        end
    else
        for kk = 1:m
            Xi = X{kk};
            col = Sp(:, kk) + Sq(:, kk);
            derivative(:, kk) = 2 * Xi * (Xi' * col)  - 2 * XY{kk};
        end
    end
    
    
    lambda_old = 1;
    sign = 1;
    
    while (sign)
        
        beta = Li / 2;
        
        hat_Sp = Sp - derivative / Li;
        hat_Sq = Sq - derivative / Li;
        
        % Compute Tp
        Tp = Solve_OneNorm(hat_Sp, beta, gamma);
        
        % Compute Tq
        [U S V] = svd(hat_Sq, 'econ');
        s_bar = diag(S);
        s_bar = s_bar( s_bar > zero_threshold );
        U = U( :, 1:length(s_bar));
        V = V( :, 1:length(s_bar));
        
        [s_hat, lambda_new ]=eplb(s_bar, size(s_bar, 1), tau, lambda_old);
        Tq = U * diag(s_hat) * V';
        
        [sign qp_term]= line_search_cond_sparse_lowrank(Tp, Tq, Sp, Sq, X, Y, derivative, Li, opts);
        
        if (sign)
            Li = Li * 2;
            lambda_old = lambda_new;
        end
        
    end
    
    funcVal = cat(1, funcVal,  qp_term +  gamma * sum( abs(Tp(:)) ) );
    
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
    t_i_min_1 = ( 1 + ( 1+4*t_i_min_1^2 )^0.5 ) / 2;
    
    Tp_i_min_1 = Tp_i;
    Tq_i_min_1 = Tq_i;
    
    Tp_i = Tp;
    Tq_i = Tq;
    
end % for loop

Tp_val = Tp;
Tq_val = Tq;
W = Tp_val + Tq_val;

end
