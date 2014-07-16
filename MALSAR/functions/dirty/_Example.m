clear;
clc;

addpath('../../utils/');

lambda1 = 1;
lambda2 = 1;


opts.tol = 10^-4;
opts.max_iter = 1000;
opts.tFlag = 1;
opts.init = 0;

sample_size_arr = [500 500 500 500 500];
task_num = length(sample_size_arr);
dimension = 200;

X = cell(task_num, 1);
Y = cell(task_num, 1);
W = rand(dimension, task_num);
for t = 1 :task_num
    % sampling
    X{t} = rand(sample_size_arr(t), dimension);
    %Y{t} = X{t} * W(:, t) + rand(sample_size_arr(t), 1) * 0.01;
    Y{t} = X{t} * W(:, t);
end


tic;
[W_1inf,fun_val,iter_1inf] = Least_Dirty(X,Y,lambda1,lambda2, opts);
toc;

plot(fun_val);


