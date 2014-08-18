%% file example_Calibration.m
%   a toy example on using different types of multi-task learning
%   calibration algorithms.
%
%% OBJECTIVE
%   see manual
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
%   Copyright (C) 2011 - 2012 Lei Yuan, Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on Aug 17, 2014.
%
%% RELATED PAPERS
%   [1] Pinghua Gong, Jiayu Zhou, Wei Fan, Jieping Ye. 
%       Efficient Multi-Task Feature Learning with Calibration. 
%       The 20th ACM SIGKDD Conference on Knowledge Discovery and 
%       Data Mining (SIGKDD 2014), New York, USA, August 24-27, 2014. 
%   [2] Han Liu, Lie Wang, Tuo Zhao. Multivariate Regression 
%       with Calibration, Technical Report. 
%
close all
clear
clc

addpath('../MALSAR/functions/calibration/')
addpath('../MALSAR/c_files/calibration/')

ni = ones(10, 1) * 100;       % sample size
m  = length(ni);              % task number 
d  = 200;                     % dimension

XSigma = diag(0.5 * ones(d, 1)) + 0.5 * ones(d, d);
sigma_max = 2*sqrt(2);

% generate models. 
zerorowratio = 0.95;
B = 20*rand(d, m) - 10;
rndnum = randperm(d);
B(rndnum(1:round(d*zerorowratio)),:) = 0;

X = cell(m, 1);
y = cell(m, 1);
for tt = 1: m
    
    % generate data matrices
    X{tt} = zeros(ni(tt), d);
    for ss = 1: ni(tt)
        X{tt}(ss, :) = mvnrnd(zeros(1, d), XSigma);
    end
    % generate targets.
    X{tt} = standardize(X{tt});
    sigma_i = 2^(- (tt-1)/4);
    y{tt} = X{tt} * B(:, tt) + randn(ni(tt), 1) * sigma_i * sigma_max;  
    fprintf('Task %u sigma: %.4f\n', tt, sigma_i * sigma_max);
end

[Xdiag, samplesize, Th_vecIdx, yvect] = diagonalize(X, y);

lambda1 = 1e-5 * sum(ni) ; % l2,1
lambda2 = 1e-5 * sum(ni); % l2

globalVerbose = 1;
globalMaxIter = 10000;
globalTol     = 1e-9;

smoothEpsilon = 1e-3;

compPrimal = @(W) primalObjD( Xdiag, yvect, Th_vecIdx, lambda1, lambda2, W );

% start value. 
initThetaMeganitude = 1;
initTheta = segL2Proj(randn(sum(ni), 1) * initThetaMeganitude, Th_vecIdx);
initW = reshape(initTheta' * Xdiag, d, m);% Compute UTh
initW = -1/lambda2*max(0,1-lambda1./repmat(sqrt(sum(initW.^2,2)),1,m)).*initW;
f0    = compPrimal(initW); % the initial function value. 


%% Optimization options.
opts_DFISTA = [];
opts_DFISTA.verbose = globalVerbose;
opts_DFISTA.maxIter = globalMaxIter;
opts_DFISTA.tol     = globalTol;
opts_DFISTA.initTheta = initTheta;

opts_DPG      = [];
opts_DPG.verbose    = globalVerbose;
opts_DPG.maxIter    = globalMaxIter;
opts_DPG.tol        = globalTol;
opts_DPG.initTheta  = initTheta;

opts_DSolver   = [];
opts_DSolver.verbose   = globalVerbose;
opts_DSolver.maxIter   = globalMaxIter;
opts_DSolver.tol       = globalTol;
opts_DSolver.initTheta = initTheta;

opts_SPG = [];
opts_SPG.verbose    = globalVerbose;
opts_SPG.maxIter    = globalMaxIter;
opts_SPG.tol        = globalTol;
opts_SPG.epsilon    = smoothEpsilon;
opts_SPG.initW      = initW;

opts_SFISTA = [];
opts_SFISTA.verbose = globalVerbose;
opts_SFISTA.maxIter = globalMaxIter;
opts_SFISTA.tol     = globalTol;
opts_SFISTA.epsilon = smoothEpsilon;
opts_SFISTA.initW   = initW;

opts_SSolver = [];
opts_SSolver.verbose = globalVerbose;
opts_SSolver.maxIter = globalMaxIter;
opts_SSolver.tol     = globalTol;
opts_SSolver.epsilon = smoothEpsilon;
opts_SSolver.initW   = initW;

opts_Subg = [];
opts_Subg.verbose   = globalVerbose;
opts_Subg.maxIter   = globalMaxIter;
opts_Subg.tol       = globalTol;
opts_Subg.initW     = initW;

opts_ADMM = [];
opts_ADMM.verbose   = globalVerbose;
opts_ADMM.maxIter   = globalMaxIter;
opts_ADMM.tol       = 1e-3;
opts_ADMM.initW     = initW;
opts_ADMM.initTheta = initTheta;

opts_CVX  = [];
opts_CVX.verbose    = globalVerbose;

%% Computation.

    methods = { @MTFLCd_DFISTA, @MTFLCd_DPG,  @MTFLCd_SFISTA, @MTFLCd_SPG, ...
                @MTFLCd_Subg,   @MTFLCd_ADMM, @MTFLC_CVX};
    opts    = { opts_DFISTA,    opts_DPG,     opts_SPG,       opts_SFISTA, ...
                opts_Subg,      opts_ADMM,    opts_CVX};


goldStdIdx = length(methods); 


legendArr = cell(length(methods), 1);
modelArr  = cell(length(methods), 1);
infoArr   = cell(length(methods), 1);
totalTimeInfo = zeros(length(methods), 1);

for i = 1: length(methods)
    totalTimer = tic;
    
    [modelArr{i}, infoArr{i}] = methods{i}( X, y, lambda1, lambda2, opts{i} );

    totalTimeInfo(i)  = toc(totalTimer);
end
opts_ADMM.initW     = initW;

%% Output 
fprintf('-------------Funcion Value INFO------------\n')
fprintf('lambda1 %.4f -- lambda2 %.4f\n', lambda1, lambda2);
for i = 1: length(methods)
    fprintf('%s \t Primal %.4f', infoArr{i}.algName, compPrimal(modelArr{i}));
    if isfield(infoArr{i}, 'fvD'), fprintf('\t Dual %.4f', infoArr{i}.fvD(end)); end
    fprintf('\n');
end
fprintf('-----------------------------------------\n')
disp(' ');
fprintf('-----------------MODEL INFO----------------\n')
W_GS = modelArr{goldStdIdx};
fprintf('Baseline method: %s\n', infoArr{goldStdIdx}.algName)
for i = 1: length(methods)
    fprintf('%s   model relative error %.4f \n', infoArr{i}.algName, norm(modelArr{i} - W_GS,'fro')/norm(W_GS,'fro'));
end
fprintf('-----------------------------------------\n\n')

%% Plot figure. 
close all;
timeInfo  = cell(length(methods), 1);
primInfo  = cell(length(methods), 1);
nameArr   = cell(length(methods), 1);
totalMethod = 0;
showLen = 50;
for i = 1: length(methods)
    if strcmp(infoArr{i}.algName, 'Primal Subg')
        continue;
    end
    if isfield(infoArr{i}, 'timeVal')
        totalMethod = totalMethod + 1;
        
        if (infoArr{i}.timeVal > showLen)
            timeInfo{totalMethod} = infoArr{i}.timeVal(1:showLen);
            primInfo{totalMethod} = infoArr{i}.fvP(1:showLen);
        else
            timeInfo{totalMethod} = infoArr{i}.timeVal;
            primInfo{totalMethod} = infoArr{i}.fvP;
        end
        
        timeInfo{totalMethod} = [0;  timeInfo{totalMethod}] + 1;
        primInfo{totalMethod} = [f0; primInfo{totalMethod}];
        nameArr{totalMethod}  = infoArr{i}.algName;
    end
end
timeInfo = timeInfo(1: totalMethod);
primInfo = primInfo(1: totalMethod);
nameArr  = nameArr(1: totalMethod);

figure;
% added by Pinghua
linecolor = {'r','g','b','c','m'};
for i = 1: totalMethod
    semilogx(timeInfo{i}, primInfo{i}, linecolor{i});
    hold on;
end
legend(nameArr);

%% Plot dual variables. 
timeInfo  = cell(length(methods), 1);
dualInfo  = cell(length(methods), 1);
nameArr   = cell(length(methods), 1);
totalMethod = 0;
showLen = 50;
for i = 1: length(methods)
    if isfield(infoArr{i}, 'timeVal') && isfield(infoArr{i}, 'fvD')
        totalMethod = totalMethod + 1;
        
        if (infoArr{i}.timeVal > showLen)
            timeInfo{totalMethod} = infoArr{i}.timeVal(1:showLen);
            dualInfo{totalMethod} = infoArr{i}.fvD(1:showLen);
        else
            timeInfo{totalMethod} = infoArr{i}.timeVal;
            dualInfo{totalMethod} = infoArr{i}.fvD;
        end
        
        timeInfo{totalMethod} = [timeInfo{totalMethod}] + 1;
        dualInfo{totalMethod} = [dualInfo{totalMethod}];
        nameArr{totalMethod}  = infoArr{i}.algName;
    end
end
timeInfo = timeInfo(1: totalMethod);
dualInfo = dualInfo(1: totalMethod);
nameArr  = nameArr(1: totalMethod);

figure;
linecolor = {'r','g','b','c','m'};
for i = 1: totalMethod
    semilogx(timeInfo{i}, dualInfo{i}, linecolor{i});
    hold on;
end
legend(nameArr);