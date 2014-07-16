function [B] = standardize(A)
% normalize A such that sum_i A_ij = 0, sum_i A_ij^2 = 1 (mean 0 and variance 1)
% each row of A is a sample

m = size(A,1);
% B = A - repmat(mean(A),m,1); % sum_i A_ij = 0
B = A;
ind = (sum(B==0) == m);
B(:,ind) = 1/(sqrt(m));
B(:,~ind) = B(:,~ind)./repmat(sqrt(sum(B(:,~ind).^2)),m,1); % sum_i A_ij^2 = 1