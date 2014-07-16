function [Y] = diagonalize(X,samplesize)
% ----------------------- Input ------------------------------
% X: samples of all tasks (each row is a sample)
% samplesize: the i-th entry is the sample size of the i-th task
% ----------------------- Output -----------------------------
% Y: sparse data matrix which is diagonal

tasknum = length(samplesize); % the number of tasks
[totalnum,dim] = size(X);
row = zeros(dim*totalnum,1); col = row; datavec = row;
accumsize = 0; accumind = 0;
for i = 1:tasknum
    accumsize = accumsize + samplesize(i);
    accumind = accumind + dim*samplesize(i);
    indsample = (accumsize-samplesize(i)+1 : accumsize)';
    indnz = (accumind-dim*samplesize(i)+1 : accumind)';
    row(indnz) = repmat(indsample,dim,1);
    col(indnz) = reshape(repmat(dim*(i-1)+1:dim*i,samplesize(i),1),dim*samplesize(i),1);
%     Xi = X(indsample,:)/sqrt(tasknum*samplesize(i));
    Xi = X(indsample,:);
    datavec(indnz) = Xi(:);
end
Y = sparse(row,col,datavec,totalnum,dim*tasknum);



