function [ W, info ] = MTFLC_CVX( X, y, lambda1, lambda2, opts ) %#ok
%MTFLC_CVX Summary of this function goes here
%   Detailed explanation goes here

if(nargin<5), opts = []; end
opts = setOptsDefault( opts, 'verbose', 1); 
verbose = opts.verbose;


info.algName = 'Primal CVX';

m = length(X);     %#ok
d = size(X{1}, 2); %#ok

if verbose <= 1
    cvx_begin quiet
    cvx_precision high
    
    variable W(d, m)
    minimize(primalObj_cvx( W, X, y, lambda1, lambda2 ) )
    cvx_end
else
    cvx_begin
    cvx_precision high
    
    variable W(d, m)
    minimize(primalObj_cvx( W, X, y, lambda1, lambda2 ) )
    cvx_end
end

info.funcVal = primalObj(W, X, y, lambda1, lambda2);
end

function [ fv ] = primalObj_cvx( W, X, y, lambda1, lambda2 )
% CVX: primal objective
fv = lambda1 * l21nrm_cvx(W) + lambda2 /2 * sum(sum(W.*W));
for i = 1: length(X)
    fv = fv + norm(X{i} * W(:, i) - y{i});
end
end

function [fv] = primalObj(W, X, y, lambda1, lambda2)
% computation of primal objective.
fv = lambda1 * sum(sqrt(sum(W.^2, 2))) + lambda2 /2 * sum(sum(W.^2));
for i = 1: length(X)
    fv = fv + sqrt(sum((X{i} * W(:, i) - y{i}).^2));
end
end

function l = l21nrm_cvx(Z)
% CVX: l2,1 norm.
l = 0;
for ii = 1: size(Z,1)
    l = l + norm(Z(ii,:));
end
end





