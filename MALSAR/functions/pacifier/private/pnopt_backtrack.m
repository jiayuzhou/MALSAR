function varargout = pnopt_backtrack( x, d, t, f_x, h_x, dg_x, smoothF, ...
  nonsmoothF, desc_param, xtol, maxIter )
% pnopt_backtrack : Backtracking line search for step that satisfies a sufficient 
%   descent condition.
% 
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
% 
% --------------------Initialize--------------------

  % Set line search parameters
  beta = 0.5;

  % Set termination flags
  FLAG_SUFFDESC = 1;
  FLAG_TOLX     = 2;
  FLAG_MAXFUNEV = 3;

  iter = 0;
  
  desc = dg_x + nonsmoothF( x + d ) - h_x;
  
  % --------------------Main Loop--------------------
  while 1
    iter = iter + 1;
    
    % Evaluate trial point and function value.
    y = x + d * t;
    if nargout > 6
      [ g_y, Dg_y, D2g_y ] = smoothF( y );
    else
      [ g_y, Dg_y ] = smoothF( y );
    end
    h_y = nonsmoothF( y );
    f_y = g_y + h_y;
    
    % Check termination criteria
    if f_y < f_x + desc_param * t * desc         % Sufficient descent condition satisfied
      flag = FLAG_SUFFDESC;  
      break
    elseif t <= xtol            % Step length too small
      flag = FLAG_TOLX;
      break
    elseif iter >= maxIter      % Too many line search iterations
      flag = FLAG_MAXFUNEV;
      break
    end

    % Backtrack if objective value not well-defined
    if isnan( f_y ) || isinf( f_y ) || abs( f_y - f_x - t * dg_x ) <= 1e-9
      t = beta * t;
    % Safeguard quadratic interpolation
    else
      t_interp = - ( dg_x * t^2) / ( 2 * ( f_y - f_x - t * dg_x ) );
      if 0.01 <= t_interp && t_interp <= 0.99*t 
        t = t_interp;
      else
        t = beta * t;
      end
    end

  end 
  
  if nargout > 6
    varargout = { y, f_y, Dg_y, D2g_y, t, flag ,iter };
  else
    varargout = { y, f_y, Dg_y, t, flag ,iter };
  end
  
  