function [Xdiag, samplesize, accIdx, yvect] = diagonalize(X, y)
% ----------------------- Input ------------------------------
% X: cell array. samples of all tasks (each row is a sample)
% y: cell array. optional. 
% 
% designed for cell arrays. 
%
% ----------------------- Output -----------------------------
% Xdiag: sparse data matrix which is diagonal
% samplesize: the i-th entry is the sample size of the i-th task
% accIdx: the start/end index of the tasks. 

tasknum = length(X); % the number of tasks
samplesize = zeros(tasknum, 1);
for i = 1:tasknum
    samplesize(i) = size(X{i}, 1);
end
totalnum = sum(samplesize);
dim = size(X{1}, 2);

% initialize the data. 
row = zeros(dim*totalnum,1); col = row; datavec = row;
accumsize = 0; accumind = 0;
accIdx = zeros(tasknum + 1, 1);
%accIdx = zeros(tasknum, 2);
accIdx(1) = 0;
for i = 1:tasknum
    %accIdx(i, :) = [ accumsize + 1, accumsize +  samplesize(i)];
    accIdx(i + 1) = accIdx(i) + samplesize(i);
    
    accumsize  = accumsize + samplesize(i);
    accumind   = accumind + dim*samplesize(i);
    indsample  = (accumsize-samplesize(i)+1 : accumsize)';
    indnz      = (accumind-dim*samplesize(i)+1 : accumind)';
    row(indnz) = repmat(indsample,dim,1);
    col(indnz) = reshape(repmat(dim*(i-1)+1:dim*i,samplesize(i),1),dim*samplesize(i),1);
    datavec(indnz) = X{i}(:);
       
end
Xdiag = sparse(row,col,datavec,totalnum,dim*tasknum);

% process label.
yvect = zeros(accumsize, 1);
if nargin >1
    for tt = 1: tasknum
        %yvect(accIdx(tt, 1): accIdx(tt, 2)) = y{tt};
        yvect(accIdx(tt) + 1: accIdx(tt+1)) = y{tt};
    end
end

