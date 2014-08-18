function op = prox_null(  )

%PROX_NNL1    null projection
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1), OP(X)>=0 
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
%
% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector
% Jiayu Zhou 2013, created NN-L1 projection. 


op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes

function v = f(x) %#ok
    v = 0; % 0 no matter what.
end

function x = prox_f(x,t)   %#ok
    % null function. directly return x. 
end


end


% TFOCS v1.2 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2012 California Institute of Technology and CVX Research.
% See the file TFOCS/license.{txt,pdf} for full license information.
