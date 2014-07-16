%% FUNCTION MultiSource_LeastR
%   Incompletet Multi-Source Learning with Least Squares Loss (optimization)
%   This function is not intended to directly called by users.
%   See the Logistic_iMSF function.
%
%% OBJECTIVE
%   see manual.
%
%% INPUT
%   A_Set: {n * d} * t - input. The cell array of data. Each cell should be
%          n by p_i, with missing samples denoted by a whole row of NaNs.
%   Y_set: {n * 1} * t - output. Y_i \in {-1, 1}.
%   z:     group sparsity regularization parameter (a relative value, [0,1])
%   G: group information, output from Construct_iMSF (see manual for details)
%   W: group information, output from Construct_iMSF (see manual for details)
%   ind: group information, output from Construct_iMSF (see manual for details)
%
%% OUTPUT
%   Sol: {struct} solution struct array.
%         For task i: model: struct{i}.x,
%   funcVal: objective function values at each iteration.
%   ValueL: line search step at each iteration.
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
%   init_opts, Construct_iMSF, Least_iMSF

function [Sol, funcVal, ValueL] = MultiSource_LeastR(A_Set, Y_Set, z, G, W, ind, opts)
%
if nargin <6
    error('\n Inputs: A_Set, Y_Set, z, G, W and ind should be specified!\n');
end

if nargin <7
    opts = [];
end

% initialize options.
opts=init_opts(opts);

% Misc
n_P = length(A_Set);
Data_Set = struct('Ind', 1:n_P);
k = size(W, 2);
gWeight=ones(k, 1);

m = 0;
% Total number of samples
n = 0;
% Total number of features

for i = 1:n_P
    Data_Set(i).A = A_Set{i};
    Data_Set(i).y = Y_Set{i};
    [Data_Set(i).m, Data_Set(i).n] = size(A_Set{i});
    m = m + Data_Set(i).m;
    n = n + Data_Set(i).n;
end


% lambda

% lambda = z;

% Here we compute lambda_max

ATy = [];

for i = 1:n_P
    % compute AT b
    Data_Set(i).ATy = Data_Set(i).A' * Data_Set(i).y;
    ATy = [ATy; Data_Set(i).ATy];
end

% compute the norm of ATb corresponding to each group
norm_ATy = zeros(k, 1);
for i = 1:k
    temp_ind = G(W(1, i):W(2, i));
    norm_ATy(i) = norm(ATy(temp_ind));
end

% incorporate the gWeight
norm_ATy = norm_ATy ./ gWeight;

% compute lambda_max
lambda_max = max(norm_ATy);

% As .rFlag=1, we set lambda as a ratio of lambda_max
lambda = z * lambda_max;

% initialize a starting point

for i = 1:n_P
    Data_Set(i).x = zeros(Data_Set(i).n, 1);
    Data_Set(i).Ax = Data_Set(i).A * Data_Set(i).x;
end

bFlag = 0; % this flag tests whether the gradient step only changes a little

L = 1 / m; % the intial guess of the Lipschitz continuous gradient

for i = 1:n_P
    % assign xp with x, and Axp with Ax
    Data_Set(i).xp = Data_Set(i).x;
    Data_Set(i).Axp = Data_Set(i).Ax;
    Data_Set(i).xxp = zeros(Data_Set(i).n,1);
end


funcVal = [];

% The Armijo Goldstein line search schemes + accelearted gradient descent

alphap=0; alpha=1;

for iter=1:opts.maxIter
    
    beta = (alphap - 1) / alpha;
    
    if opts.pFlag
        parfor i = 1:n_P
            % --------------------------- step 1 ---------------------------
            % compute search point s based on xp and x (with beta)
            Data_Set(i).s = Data_Set(i).x + beta * Data_Set(i).xxp;
            
            % --------------------------- step 2 ---------------------------
            % line search for L and compute the new approximate solution x
            
            % compute As=A*s
            Data_Set(i).As = Data_Set(i).Ax + beta * (Data_Set(i).Ax - Data_Set(i).Axp);
            
            % aa= - diag(y) * (A * s + sc)
            %         Data_Set(i).aa = - Data_Set(i).y .* (Data_Set(i).As + Data_Set(i).sc);
            
            % fun_s is the least square loss at the search point
            Data_Set(i).fun_s = 0.5 * norm(Data_Set(i).As - Data_Set(i).y) ^ 2;
            
            % compute g= AT As - ATy, the gradient of s
            Data_Set(i).g = Data_Set(i).A' * Data_Set(i).As - Data_Set(i).ATy;
            
            % copy x and Ax to xp and Axp
            Data_Set(i).xp = Data_Set(i).x;
            Data_Set(i).Axp = Data_Set(i).Ax;
        end
    else
        for i = 1:n_P
            % --------------------------- step 1 ---------------------------
            % compute search point s based on xp and x (with beta)
            Data_Set(i).s = Data_Set(i).x + beta * Data_Set(i).xxp;
            
            % --------------------------- step 2 ---------------------------
            % line search for L and compute the new approximate solution x
            
            % compute As=A*s
            Data_Set(i).As = Data_Set(i).Ax + beta * (Data_Set(i).Ax - Data_Set(i).Axp);
            
            % aa= - diag(y) * (A * s + sc)
            %         Data_Set(i).aa = - Data_Set(i).y .* (Data_Set(i).As + Data_Set(i).sc);
            
            % fun_s is the least square loss at the search point
            Data_Set(i).fun_s = 0.5 * norm(Data_Set(i).As - Data_Set(i).y) ^ 2;
            
            % compute g= AT As - ATy, the gradient of s
            Data_Set(i).g = Data_Set(i).A' * Data_Set(i).As - Data_Set(i).ATy;
            
            % copy x and Ax to xp and Axp
            Data_Set(i).xp = Data_Set(i).x;
            Data_Set(i).Axp = Data_Set(i).Ax;
        end
    end
    
    % Combine the variables from the sub-problems
    fun_s = 0;
    g = [];
    s = [];
    xp = [];
    for i = 1:n_P
        fun_s = fun_s + Data_Set(i).fun_s;
        g = [g; Data_Set(i).g];
        s = [s; Data_Set(i).s];
        xp = [xp; Data_Set(i).xp];
    end
    
    while (1)
        
        % let s walk in a step in the antigradient of s to get v
        % and then do the L1/Lq-norm regularized projection
        v = s - g / L;
        
        % L1/L2-norm regularized projection
        x = l1_l2(v, W, G, k, n, lambda/ L * gWeight);
        
        v = x - s;  % the difference between the new approximate solution x
        % and the search point s
        
        % Decompose the new solution to each sub-problems
        for i = 1:n_P
            Data_Set(i).x = x(ind(i) + 1:ind(i + 1));
        end
        
        fun_x = 0;
        
        if opts.pFlag
            parfor i = 1:n_P
                % compute A x
                Data_Set(i).Ax = Data_Set(i).A * Data_Set(i).x;
                
                % fun_x is the least square loss at the new approximate solution
                Data_Set(i).fun_x = 0.5 * norm(Data_Set(i).Ax - Data_Set(i).y) ^ 2;
                
                fun_x = fun_x + Data_Set(i).fun_x;
            end
        else
            for i = 1:n_P
                % compute A x
                Data_Set(i).Ax = Data_Set(i).A * Data_Set(i).x;
                
                % fun_x is the least square loss at the new approximate solution
                Data_Set(i).fun_x = 0.5 * norm(Data_Set(i).Ax - Data_Set(i).y) ^ 2;
                
                fun_x = fun_x + Data_Set(i).fun_x;
            end
        end
        
        r_sum = (v' * v) / 2;
        l_sum = fun_x - fun_s - v' * g;
        
        if (r_sum <= 1e-20)
            bFlag = 1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        % the condition is fun_x <= fun_s + v'* g + c * gc
        %                           + L/2 * (v'*v + (c-sc)^2 )
        if(l_sum <= r_sum * L)
            break;
        else
            L = max(2 * L, l_sum / r_sum);
            %fprintf('\n L=%e, r_sum=%e',L, r_sum);
        end
    end
    
    % --------------------------- step 3 ---------------------------
    % update alpha and alphap, and check whether converge
    alphap = alpha;
    alpha = (1 + sqrt(4 * alpha * alpha + 1)) / 2;
    
    ValueL(iter)=L;
    % store values for L
    
    xxp = x - xp;
    % Decompose the new results to each sub-problems
    for i = 1:n_P
        Data_Set(i).xxp = xxp(ind(i) + 1:ind(i + 1));
    end
    
    %funVal(iterStep)=fun_x;
    
    % the q-norm of x
    norm_x_k = zeros(k, 1);
    for i = 1:k
        norm_x_k(i) = norm(x(G(W(1, i):W(2, i))));
    end
    
    % function value = loss + regularizatioin
    %funVal(iterStep)=fun_x + lambda * norm_x_k'* gWeight;
    funcVal = cat(1, funcVal,fun_x + lambda * norm_x_k'* gWeight);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
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
end

Sol = cell(n_P, 1);
for i = 1:n_P
    Sol{i}.x = x(ind(i) + 1:ind(i + 1));
end

function x = l1_l2(v, W, G, k, n, lambda)
% l1-l2 projection
%
x = v;

for i = 1:k
    ind = G(W(1, i):W(2, i));
    temp = v(ind);
    norm_temp = norm(temp);
    x(ind) = (max(norm_temp - lambda(i), 0) / norm_temp) * temp;
end