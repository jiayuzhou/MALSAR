%% FUNCTION Construct_iMSF
%   Prepares input for iMSF optimization. This function is not intended to 
%   directly called by users. See the Logistic_iMSF function.
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

function [A_Set, Y_Set, W, G, ind, PS, A_Set_Complete] = Construct_iMSF(X_Set, Y)
% 

X = [];
temp = [];
IndMissing = [];
nF = [];
FeatureTypes = cell(length(X_Set), 1);

for i = 1:length(X_Set)
    temp = X_Set{i};
    FeatureTypes{i} = sprintf('%d', i);
    
    temp(~isnan(temp(:, 1)), :) = zscore(temp(~isnan(temp(:, 1)), :));
    
    IndMissing = [IndMissing ~isnan(temp(:, 1))];
    X = [X temp];
    nF = [nF size(temp, 2)];
end

IndAllZero = sum(IndMissing, 2) == 0;
IndMissing = IndMissing(~IndAllZero, :);
numIndMissing = bin2dec(num2str(IndMissing));
% IndMissing denotes what type of feature is missing for each particular
% sample.

X = X(~IndAllZero, :);
Y = Y(~IndAllZero, :);

PS = nchoose(FeatureTypes);
% PS is the collection of all combinations.
ind_valid = true(length(PS), 1);

% The starting index of each feature type.
nF_Start = [0 cumsum(nF)];
nF_Start = nF_Start(1:end - 1) + 1;

% We shall first remove the combinations that have too few samples.
for i = 1:length(PS)
    curTypes = PS{i};
    temp = zeros(1, length(FeatureTypes));
    
    for j = 1:length(curTypes)
        type = curTypes{j};
        temp(strcmp(type, FeatureTypes)) = 1;
    end
    
    temp = bin2dec(num2str(temp));
    
    ind_temp = numIndMissing == temp;
    Y_Cur = Y(ind_temp);
    
    if length(unique(Y_Cur)) < 2
        % If there is not enough training label (at least 1 pos 1 neg), we
        % remove this combination.
        ind_valid(i) = false;
    end
end

PS = PS(ind_valid);

IndMatrix = zeros(length(PS), sum(nF));

IndStart = 1;

for i = 1:length(PS)
    curTypes = PS{i};
    
    for j = 1:length(curTypes)
        type = curTypes{j};
        loc_type = find(strcmp(type, FeatureTypes));
        IndCur = nF_Start(loc_type):nF_Start(loc_type) + nF(loc_type) - 1;
        IndMatrix(i, IndCur) = IndStart:IndStart + nF(loc_type) - 1;
        IndStart = IndStart + nF(loc_type);
    end
end

W = [];
G = [];
IndStart = 1;

for i = 1:size(IndMatrix, 2)
    curGroup = IndMatrix(:, i)';
    curGroup(curGroup == 0) = [];
    G = [G curGroup];
    W = [W [IndStart; IndStart + length(curGroup) - 1]];
    IndStart = IndStart + length(curGroup);
end

A_Set = cell(length(PS), 1);
A_Set_Complete = cell(length(PS), 1);
Y_Set = cell(length(PS), 1);
ind = zeros(length(PS) + 1, 1);

for i = 1:length(PS)
    curTypes = PS{i};
    temp = zeros(1, length(FeatureTypes));
    ind_col = false(1, sum(nF));
    
    for j = 1:length(curTypes)
        type = curTypes{j};
        loc_type = find(strcmp(type, FeatureTypes));
        temp(loc_type) = 1;
        IndCur = nF_Start(loc_type):nF_Start(loc_type) + nF(loc_type) - 1;
        ind_col(IndCur) = true;
    end
    
    ind(i + 1) = ind(i) + nnz(ind_col);
    
    temp = bin2dec(num2str(temp));
    
    ind_temp = numIndMissing == temp;
    
    A_Set{i} = X(ind_temp, ind_col);
    A_Set_Complete{i} = X(ind_temp, :);
    Y_Set{i} = Y(ind_temp);
end