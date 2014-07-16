%% FUNCTION bsa_ihb
%    Singular Projection 
%
%% OBJECTIVE
%   min 1/2*||x - a||_2^2
%    s.t. b'*x = r, 0<= x <= u,  b > 0
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Pinghua Gong and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% Related papers
%
% [1] KC. Kiwiel. On linear-time algorithms for the continuous 
%     quadratic knapsack problem, Journal of Optimization Theory 
%     and Applications, 2007
%


function [x_star,t_star,iter] = bsa_ihb(a,b,r,u)

% initilization
break_flag = 0;
t_l = a./b; t_u = (a - u)./b;
T = [t_l;t_u];
t_L = -inf; t_U = inf;
g_tL = 0; g_tU = 0;

iter = 0;
while ~isempty(T)
    iter = iter + 1;
    g_t = 0;
    t_hat = median(T);  
    
    U = t_hat < t_u;
    M = (t_u <= t_hat) & (t_hat <= t_l); 

    if sum(U)
       g_t = g_t + b(U)'*u(U); 
    end
    if sum(M)
        g_t = g_t + sum(b(M).*(a(M) - t_hat*b(M)));
    end
    
    if g_t > r
        t_L = t_hat;
        T = T(T > t_hat);
        g_tL = g_t;
    elseif g_t < r
        t_U = t_hat;
        T = T(T < t_hat);
        g_tU = g_t;
    else
        t_star = t_hat;
        break_flag = 1;
        break;            
    end
end
if ~break_flag
     t_star = t_L - (g_tL -r)*(t_U - t_L)/(g_tU - g_tL);     
end
x_star = min(max(0,a - t_star*b),u);
end