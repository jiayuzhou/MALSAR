function [W,fun,iter] = wLassomtl(X,y,XtX,Xty,lambda,W0,tol,maxiter,L,Lflag)
% Lasso Multi-Task Learning
% min_W ||XW - Y||_F^2 + \sum_i lambda_ij*|W^ij|
% By Nesterov's method
% ------------------------------- Input ---------------------------------------------
% X: block diagonal data matrix whose i-th block (n_i*d) is the data matrix of the i-th task
% y: \sum_i{n_i}*1 vector; response vector which is stacked by the reponses of all tasks
% XtX: X'*X
% Xty: X'*y
% lambda: regularized parameter (vector)
% W0: d*m matrix; starting point of W 
% tol: stopping tolerance
% maxiter: maximum iterative steps
% Lflag: estimate the upper bound of Lipschitz constant if nonzero, zero otherwise 
%
% ------------------------------- Output -------------------------------------------
%
% W: output weight
% fun: function values
% iter: iterative steps 
%
% -----------------------------------------------------------------------------------

W = W0;
[d,m] = size(W); % d: dimension, m: the number of tasks

Wn = W;
t_new = 1; 
fun = zeros(maxiter+1,1);

% Initial function value
fun(1) = norm(X*W(:) - y)^2 + wL1norm(W,lambda);
count = 0;
for iter = 1:maxiter
    W_old = W;
    t_old = t_new;
    gradvec = 2*(XtX*Wn(:) - Xty);
    gradmat = reshape(gradvec,d,m);
    % If we estimate the upper bound of Lipschitz constant, no line search
    % is needed.
    if Lflag
        W = proximalwL1norm(Wn - gradmat/L, lambda/L);
    else
        % line search 
        for inneriter = 1:20
            W = proximalwL1norm(Wn - gradmat/L, lambda/L);
            dW = W - Wn;
            if 2*(dW(:)'*XtX*dW(:)) <= L*sum(sum((dW.*dW)))
                break;
            else
                L = L*2;
            end
        end
    end
    fun(iter+1) = norm(X*W(:) - y)^2 + wL1norm(W,lambda);
    % stopping condition
    if abs(fun(iter) - fun(iter+1))/fun(iter+1) < tol
        count = count + 1;
    else
        count = 0;
    end
    if count >= 1
        break;
    end

    % Update the coefficient
    t_new = (1+sqrt(1+4*t_old^2))/2;
    Wn = W + (t_old-1)/t_new*(W - W_old);
    
end


