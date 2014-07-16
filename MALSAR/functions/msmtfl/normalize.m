function [X] = normalize(X,samplesize)
% ----------------------- Input ------------------------------
% X: samples of all tasks (each row is a sample)
% samplesize: the i-th entry is the sample size of the i-th task
% ----------------------- Output -----------------------------
% X: normalized data matrix

tasknum = length(samplesize); % the number of tasks
accumsize = 0;
for i = 1:tasknum
    accumsize = accumsize + samplesize(i);
    indsample = (accumsize-samplesize(i)+1 : accumsize)';
    X(indsample,:) = standardize(X(indsample,:)); % standardization
%     X(indsample,:) = mapstd(X(indsample,:)')'; % standardization
end

