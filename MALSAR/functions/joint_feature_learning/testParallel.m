addpath('../../utils/')

clear;
clc;
close;

rng('default')

opts.tol = 10^-5;
opts.maxIter = 1000;
opts.tFlag = 1;
opts.init = 0;
%opts.pFlag = true;


%% Least Squares Loss

% large test
d_arr = [10000, 20000, 30000, 40000, 50000];
n_arr = [5000, 5000, 5000, 5000, 5000];

for e_idx = 1: length(d_arr)
    
    d = d_arr(e_idx);
    n = n_arr(e_idx);
    t = 12;
    
    X = cell(t, 1);
    Y = cell(t, 1);
    
    for i = 1: t
        X{i} = rand(n, d);
        Y{i} = sign(randn(n, 1)*5);
    end
    
    %rho1 = 30;  % rho3: L2,1-norm group Lasso parameter.
    
    %profile on
    %[w funcVal] = Least_L21(X, Y, rho1, opts);
    
    rho1 = 0.02;  % rho3: L2,1-norm group Lasso parameter.
    %[w c funcVal] = Logistic_L21(X, Y, rho1, opts);
    %fprintf('Zero rows: %u\n', nnz(sum(w,2 )==0));
    %fprintf('Zero row percentage: %.4g\n', nnz(sum(w,2 )==0)/d);
    %profile off
    %profile viewer
    %plot(funcVal);
    
    tic;
    opts.pFlag = true;
    [w2 c2 funcVal2] = Logistic_L21(X, Y, rho1, opts);
    toc;
    
    tic;
    opts.pFlag = false;
    [w1 c1 funcVal1] = Logistic_L21(X, Y, rho1, opts);
    toc;
    
end