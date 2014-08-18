function [U, solver_output] = solveU_ind(U0, V0, Ss, reg_l1, solver_options)
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

%non_smooth = prox_l1(reg_l1);     % regular l1.
%non_smooth = prox_null();          % no regularization.


n = length(U0);
U = cell(n, 1);
sizeU = size(U0{1});

for i = 1: n
    % note that Us = U0, Vs = V0.
    smoothF = @(Ui) evalFvGd (reshape(Ui, sizeU), V0{i}, U0{i}, V0{i}, Ss{i}); % smooth part.
    non_smooth = prox_nnl1(reg_l1);     % non-negative l1.
    
    [ Ui, ~, solver_output ] = solver(smoothF, non_smooth, U0{i}(:), solver_options);
    U{i} = reshape(Ui, sizeU); % transform to vector shape.
end

end



function [fv, gd_vect] = evalFvGd(Ui, Vi, Usi, Vsi, Ssi)

fv = funcVal(Ui, Vi, Usi, Vsi, Ssi);
if nargout>1
    gd_vect = gradVal(Ui, Vi, Usi, Vsi, Ssi);
    gd_vect = gd_vect(:);
end

end

function fv = funcVal(Ui, Vi, Usi, Vsi, Ssi)
% compute function.
funcVal_unit =@(A, B, P, U, V) ...
    sum(sum(B.*((A' * A) * B))) + 2 * sum(sum((A' * P).* B)) ...
    + sum(sum(P.* P)) - 2 * sum(sum(B .* ((A' * U) * V))) ...
    - 2* sum(sum((U' * P) .* V)) + sum(sum(V .* ((U' * U) * V)));

ti = size(Ssi, 2);
fv = funcVal_unit(Usi, Vsi, Ssi, Ui, Vi)/ ti /2;

end

function grad = gradVal(Ui, Vi, Usi, Vsi, Ssi)
% compute gradient.
ti = size(Ssi, 2);
grad = (Ui * (Vi * Vi') - (Usi * (Vsi * Vi') + Ssi * Vi'))/ti;
end