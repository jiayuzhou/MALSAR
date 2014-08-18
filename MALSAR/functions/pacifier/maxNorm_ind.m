function diffNm = maxNorm_ind(V, V0, U, U0)
n = length(V0);
diffV = zeros(n, 1);
diffU = zeros(n, 1);
for i = 1: n
    diffV(i) = norm(V{i} - V0{i}, 'fro');
    diffU(i) = norm(U{i} - U0{i}, 'fro');
end
diffNm = max(mean(diffV), mean(diffU));

end