%% FUNCTION LSSMTC
%   Learning the Shared Subspace for Multi-Task Clustering
%
%% OBJECTIVE
%   min lambda*sum_k||X_k-M_k*P_k||^2+lambda*||W'*X_k-M*P_k||^2
%       s.t. P_k>=0, W'W=I, W>=0;
%
%% INPUT
%   cellX:  cell array of dxn Data matrices, n is the number of documents,
%          d is the number of features
%   cellgnd: cell array of ground truth.
%   c: cluster number.
%   l: Dimensionality of the shared subspace
%   lambda: Regularization parameter
%   ITE: number of iterations
%
%% OUTPUT
%   W: dxl Projection matrix (shared subspace)
%   cellM: cell array of cluster centers.
%   cellP: cell array of cluster assignment.
%   residue: residue
%   cellAcc: accuracy of each task
%   cellNMI: normalized mutual information for each task.
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
%   Copyright (C) 2011 - 2012 Quanquan Gu, Jiayu Zhou, and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 17, 2012.
%
%% RELATED PAPERS
%   [1]Quanquan Gu, Jie Zhou. Learning the shared subspace
%      for multi-task clustering and transductive transfer
%      classification. In Proc. of the International Conference
%      on Data Mining (ICDM), Miami, Florida, USA, 2009.

function [W cellM cellP funcVal cellAcc cellNMI] = LSSMTC(cellX,cellgnd,c,l,lambda, opts)

% initialize options.
opts=init_opts(opts);

computeMetric = nargout > 4; % compute accuracy, NMI if necessary.

if computeMetric && isempty(cellgnd)
    error('Need groud truth variable if accuracy and/or NMI are needed.')
end

%number of tasks
m = length(cellX);

cellM = cell(m,1);
cellP = cell(m,1);
cellAcc = cell(m,1);
cellNMI = cell(m,1);

%initionalization of P_k
for i = 1:m
    [d nm] = size(cellX{i});
    cellP{i} = abs(rand(nm,c));
end


eps=1e-9; % set your own tolerance

%initionalization of W
W = eye(d,l);

X = [];
P = [];
for k=1:m
    X = [X cellX{k}];
    P = [P; cellP{k}];
end

funcVal = [];
for iter = 1:opts.maxIter
    
    
    %update in M
    M = W'*X*P*inv(P'*P);
    
    
    if opts.pFlag
        parfor k = 1:m
            %update in M_k
            cellM{k} = cellX{k}*cellP{k}*inv(cellP{k}'*cellP{k});
            
            A = lambda*cellX{k}'*cellM{k}+(1-lambda)*cellX{k}'*W*M;
            B = lambda*cellM{k}'*cellM{k}+(1-lambda)*M'*M;
            
            PA = (abs(A) + A)/2;
            NA = (abs(A) - A)/2;
            PB = (abs(B) + B)/2;
            NB = (abs(B) - B)/2;
            
            %multiplicative update in P_k
            cellP{k} = cellP{k}.*sqrt((PA + cellP{k}*NB)./(NA + cellP{k}*PB + eps));
        end
    else
        for k = 1:m
            %update in M_k
            cellM{k} = cellX{k}*cellP{k}*inv(cellP{k}'*cellP{k});
            
            A = lambda*cellX{k}'*cellM{k}+(1-lambda)*cellX{k}'*W*M;
            B = lambda*cellM{k}'*cellM{k}+(1-lambda)*M'*M;
            
            PA = (abs(A) + A)/2;
            NA = (abs(A) - A)/2;
            PB = (abs(B) + B)/2;
            NB = (abs(B) - B)/2;
            
            %multiplicative update in P_k
            cellP{k} = cellP{k}.*sqrt((PA + cellP{k}*NB)./(NA + cellP{k}*PB + eps));
        end
    end
    
    P = [];
    for k=1:m
        P = [P; cellP{k}];
    end
    
    C = X*X';
    D = X*P*M';
    PC = (abs(C) + C)/2;
    NC = (abs(C) - C)/2;
    PD = (abs(D) + D)/2;
    ND = (abs(D) - D)/2;
    tmp = W*W';
    %multiplicative update in P_k
    W = W.*sqrt((NC*W + tmp*PC*W+PD+tmp*ND)./(PC*W + tmp*NC*W+ND+tmp*PD + eps));
    
    %calculate the objective function value
    tmp1 = 0;
    for k = 1:m
        tmp1 = tmp1 + lambda*sum(sum((cellX{k}-cellM{k}*cellP{k}').^2))+(1-lambda)*sum(sum((W'*cellX{k}-M*cellP{k}').^2));
    end
    funcVal = [funcVal tmp1];
    
    %debug
    if computeMetric == 1
        for i = 1:m
            res = matrix2res(cellP{i});
            res = bestMap(cellgnd{i},res);
            tmp2 = length(find(cellgnd{i} == res))/length(cellgnd{i})*100;
            cellAcc{i} = [cellAcc{i} tmp2];
            tmp3 =MutualInfo(cellgnd{i},res)*100;
            cellNMI{i} = [cellNMI{i} tmp3];
        end
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
