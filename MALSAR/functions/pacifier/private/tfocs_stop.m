function stop = tfocs_stop( x, nonsmoothF, optTol ) 
% tfocs_stop : TFOCS stopping condition
%
%   $Revision: 0.1.2 $  $Date: 2012/09/15 $
% 
  global subprob_Dg_y subprob_optim
  
  [ ~, x_prox ]   = nonsmoothF( x - subprob_Dg_y ,1);
    subprob_optim = norm( x_prox - x ,'inf');
    stop          = subprob_optim <= optTol;
  
    