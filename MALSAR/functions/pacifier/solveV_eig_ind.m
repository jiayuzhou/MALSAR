function V = solveV_eig_ind(U, Us, Vs, Ss, reg_smooth, reg_l2)
% solve S analytically using eigen-decomposition
%
% TODO: check with optimization form.

n = length(Ss);
V = cell(n, 1);

for ii = 1:n
    UtU = U{ii}' * U{ii};
    [Q1, Lu] = eig(UtU);    
    V{ii} = solveVi(U{ii}, Us{ii}, Q1, diag(Lu) + reg_l2, Vs{ii}, Ss{ii}, reg_smooth);
end

end

function Vi = solveVi (U, Us, Q1, Lu, Vsi, Ssi,  reg_smooth)

ti = size(Ssi, 2);

iy  = [1: (ti - 1), 1: (ti - 1)];
ix  = [1: (ti - 1), 2: ti];
val = [ones(1, ti-1), -1 * ones(1, ti-1)];
Ri  = sparse(ix, iy, val, ti, ti-1);

%[Q2, Lr] = eig(reg_smooth * (Ri * Ri'), ti);
[Q2, Lr] = eig(reg_smooth * full(Ri * Ri'));
Lr = diag(Lr);

UtSi = Q1' * (U' * Us) * Vsi * Q2 + Q1' * (U' * Ssi) * Q2;

V_hat = zeros(size(UtSi));

for i = 1:size(V_hat, 1)
    for j = 1: size(V_hat, 2)
        V_hat(i, j) = UtSi(i, j) / (Lu(i) + Lr(j));
    end
end

Vi = Q1 * V_hat * Q2';

end