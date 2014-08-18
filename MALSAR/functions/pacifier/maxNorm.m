function diffNm = maxNorm(V, V0, U, U0)
n = length(V0);
diffV = zeros(n, 1);
for i = 1: n
    diffV(i) = norm(V{i} - V0{i}, 'fro');
end
diffNm = max(mean(diffV), norm(U - U0, 'fro'));

end