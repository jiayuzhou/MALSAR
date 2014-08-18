function [ f_y, Df_y ] = pnopt_quad( P, q, r, x )
% pnopt_quad : PNOPT local quadratic approximation
%
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
% 
  global subprob_Dg_y
  
  if isa( P, 'function_handle' )
    H_x = P( x );
  else
    H_x = P * x;
  end
  
  f_y = 0.5 * x' * H_x + q' * x + r;
  if nargout > 1
    subprob_Dg_y = H_x + q;
    Df_y = subprob_Dg_y;
  end