function [U, solver_output] = solveU(U0, V0, Ss, reg_l1, solver_options)
% solve U using non-negative l1 regularized regression.

solver = @pnopt_sparsa;
if nargin<5
    solver_options = pnopt_optimset(...
        'display'   , 0    ,...
        'maxfunEv'  , 5000 ,...
        'maxIter'   , 500  ,...
        'ftol'      , 1e-9 ,...
        'optim_tol' , 1e-6 ,...
        'xtol'      , 1e-9 ...
        );
end

% note that Us = U0, Vs = V0.
smoothF = @(U) evalFvGd (reshape(U, size(U0)), V0, U0, V0, Ss); % smooth part.
non_smooth = prox_nnl1(reg_l1);     % non-negative l1.
%non_smooth = prox_l1(reg_l1);     % regular l1.
%non_smooth = prox_null();          % no regularization.

[ U, ~, solver_output ] = solver(smoothF, non_smooth, U0(:), solver_options);
U = reshape(U, size(U0)); % transform to vector shape.

end



function [fv, gd_vect] = evalFvGd(U, V_cell, Us, Vs, Ss)

fv = funcVal(U, V_cell, Us, Vs, Ss);
if nargout>1
    gd_vect = gradVal(U, V_cell, Us, Vs, Ss);
    gd_vect = gd_vect(:);
end

end

function fv = funcVal(U, V, Us, Vs, Ss)
% compute function.
funcVal_unit =@(A, B, P, U, V) ...
    sum(sum(B.*((A' * A) * B))) + 2 * sum(sum((A' * P).* B)) ...
    + sum(sum(P.* P)) - 2 * sum(sum(B .* ((A' * U) * V))) ...
    - 2* sum(sum((U' * P) .* V)) + sum(sum(V .* ((U' * U) * V)));

fv = 0;
for nn = 1: length(Ss)
    ti = size(Ss{nn}, 2);
    lossVal = funcVal_unit(Us, Vs{nn}, Ss{nn}, U, V{nn})/ ti /2;
    fv = fv + lossVal;
end

end

function grad = gradVal(U, V, Us, Vs, Ss)
% compute gradient.
grad = zeros(size(U));
for nn = 1: length(Ss)
    ti = size(Ss{nn}, 2);
    grad = grad + (U * (V{nn} * V{nn}') ...
        - (Us * (Vs{nn} * V{nn}') + Ss{nn} * V{nn}'))/ti;
end

end