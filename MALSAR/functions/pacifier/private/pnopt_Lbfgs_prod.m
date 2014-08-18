function H_x = pnopt_Lbfgs_prod( s_old, y_old, de ) 
% pnopt_Lbfgs_prod : Product with L-BFGS Hessian approximation
% 
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
% 
  l = size( s_old, 2 );
  L = zeros( l );
  for k = 1:l;
    L(k+1:l,k) = s_old(:,k+1:l)' * y_old(:,k);
  end
  d1 = sum( s_old .* y_old );
  d2 = sqrt( d1 );
  
% %   Mark Schmidt's code (slow)
%   Qty1 = et * ( sPrev' * sPrev );
%   Qty2 = [ et  *sPrev, yPrev ];
%   Qty3 = [ Qty1, L; L', - diag(d1) ];
%   Hx   = @(x) et  *x - Qty2 * ( Qty3 \ ( Qty2' *x ) );
%   
  R    = chol( de * ( s_old' * s_old ) + L * ( diag( 1 ./ d1 ) * L' ), 'lower' );
  R1   = [ diag( d2 ), zeros(l); - L*diag( 1 ./ d2 ), R ];
  R2   = [- diag( d2 ), diag( 1 ./ d2 ) * L'; zeros( l ), R' ];
  Qty2 = [ y_old, de * s_old ];
  H_x = @(x) de * x - Qty2 * ( R2 \ ( R1 \ ( Qty2' * x ) ) );
  