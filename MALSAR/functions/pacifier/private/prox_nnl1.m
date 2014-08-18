function op = prox_nnl1( q )

%PROX_NNL1    non-negative L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1), OP(X)>=0 
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
%
% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector
% Jiayu Zhou 2013, created NN-L1 projection. 


if nargin == 0,
	q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) %|| all(q==0) %|| numel( q ) ~= 1
	error( 'Argument must be positive.' );
end

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes

function v = f(x)
    v = norm( q(:).*x(:), 1 );
end

function x = prox_f(x,t)  
    tq = t .* q; % March 2012, allowing vectorized stepsizes
    %x  = x .* 1 - min( tq./abs(x), 1 ); traditional 
    x  = max(x - tq, 0);
end


end


% TFOCS v1.2 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2012 California Institute of Technology and CVX Research.
% See the file TFOCS/license.{txt,pdf} for full license information.
