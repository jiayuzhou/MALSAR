
%%
%nested cross validation for testing the logistic rMTFL model
%The stratified cv are integrated into method to handle
%"not_enough_data_sample" problems and privide more accurate estimator
%Han Cao
%24.02.2017
%%

clear;
clc;
close;

addpath('../MALSAR/functions/rMTFL/'); % load function
addpath('../MALSAR/utils/'); % load utilities
addpath('./train_and_test/'); 


% simulate the data
n = 50;
d = 300;
T = 10;

X = cell(T, 1);
Y = cell(T, 1);
W = randn(d, T);
W_mask = abs(randn(d, T))<1;
W(W_mask) = 0;
for i = 1: T
    X{i} = randn(n, d);
    Y{i} = sign(X{i} * W(:, i) + rand(n, 1) * 0.01);
end



%optimization options
opts.init = 2;  
opts.tFlag = 1; 
opts.tol = 10^-5;
opts.maxIter = 60000; 

% lambda range
lambda1_range = [1:-0.01:0.01];
lambda2_range = [2:-0.05:0.05];

%container for holding the results
r_acc=cell(1,3);
r_inCvAcc=cell(1,3); %
r_S=cell(1,3);


%nested cross validation
out_cv_fold=3;
in_cv_fold=5;
for i = 1: out_cv_fold
    Xtr = cell(T, 1);
    Ytr = cell(T, 1);
    Xte = cell(T, 1);
    Yte = cell(T, 1);
    
    %stratified cross validation
    for t = 1: T
        task_sample_size = length(Y{t});
        ct = find(Y{t}<0);
        cs = find(Y{t}>0);
        ct_idx = i : out_cv_fold : length(ct);
        cs_idx = i : out_cv_fold : length(cs);
        te_idx = [ct(ct_idx); cs(cs_idx)];
        tr_idx = setdiff(1:task_sample_size, te_idx);
        
        Xtr{t} = X{t}(tr_idx, :);
        Ytr{t} = Y{t}(tr_idx, :);
        Xte{t} = X{t}(te_idx, :);
        Yte{t} = Y{t}(te_idx, :);
    end
    
    %inner cv
    fprintf('inner CV started\n')
    [best_lambda1 best_lambda2 accuracy_mat] = CrossValidationDirty( Xtr, Ytr, ...
        'Logistic_rMTFL', opts, lambda1_range,lambda2_range, in_cv_fold, ...
        'eval_MTL_accuracy');
    
    %train
    %warm start for one turn
    [W C P Q L F] = Logistic_rMTFL(Xtr, Ytr, best_lambda1, best_lambda2, opts);
    opts2=opts;
    opts2.init=1;
    opts2.C0=C;
    opts2.P0=P;
    opts2.Q0=Q;
    opts2.tol = 10^-10;
    [W2 C2 P2 Q2 L2 F2] = Logistic_rMTFL(Xtr, Ytr, best_lambda1, best_lambda2, opts2);
    

     %test
    final_performance = eval_MTL_accuracy(Yte, Xte, W2, C2);
    
    %collect results
    r_acc{i}=final_performance;
    r_inCvAcc{i}=accuracy_mat;
    r_S{i}=nnz((sum(P2,2)==0))/size(P2,1);
  
end

 fprintf('the average accuracy is \n')
 disp(mean(cell2mat(r_acc)))
 
 %cv accuracy cross lambda
for i=1:out_cv_fold
    surf(lambda1_range', lambda2_range,r_inCvAcc{i}' );
    xlabel('Parameter for P');
    ylabel('parameter for Q');
    hold on; 
end
hold off;
title('cross validation accuracy over different lambda');
set(gca,'FontSize',12);
print('-dpdf', '-r100', 'LogisticDirty');

 
