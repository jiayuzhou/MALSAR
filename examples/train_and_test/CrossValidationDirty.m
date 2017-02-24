
%%
%cross validation for composite model: W=P+Q
%Han Cao
%24.02.2017
%%


function [ best_lambda1 best_lambda2 perform_mat] = CrossValidationDirty...
    ( X, Y, obj_func_str, obj_func_opts, lambda1_range, lambda2_range, cv_fold, eval_func_str)



eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);


% compute sample size for each task
task_num = length(X);

% performance vector
perform_mat = zeros(length(lambda1_range),length(lambda2_range))';
perform_mat = mat2cell(perform_mat,size(perform_mat,1),ones(1,size(perform_mat,2)));



% begin cross validation
fprintf('[')
for cv_idx = 1: cv_fold
    fprintf('.')
    
    % buid cross validation data splittings for each task.
    cv_Xtr = cell(task_num, 1);
    cv_Ytr = cell(task_num, 1);
    cv_Xte = cell(task_num, 1);
    cv_Yte = cell(task_num, 1);
    
    
    %stratified cross validation
    for t = 1: task_num
        task_sample_size = length(Y{t});

        ct = find(Y{t}<0);
        cs = find(Y{t}>0);
        ct_idx = cv_idx : cv_fold : length(ct);
        cs_idx = cv_idx : cv_fold : length(cs);
        te_idx = [ct(ct_idx); cs(cs_idx)];
        tr_idx = setdiff(1:task_sample_size, te_idx);
        
        cv_Xtr{t} = X{t}(tr_idx, :);
        cv_Ytr{t} = Y{t}(tr_idx, :);
        cv_Xte{t} = X{t}(te_idx, :);
        cv_Yte{t} = Y{t}(te_idx, :);
    end

    %cv with warm start
    parfor i= 1: length(lambda1_range)
        opts=obj_func_opts;
        opts.init=2;
        if(isfield(opts, 'P0'))
            opts = rmfield(opts, 'P0');
        end
        if(isfield(opts, 'Q0'))
            opts = rmfield(opts, 'Q0');
        end
        if(isfield(opts, 'C0'))
            opts = rmfield(opts, 'C0');
        end
        for ii= 1: length(lambda2_range)
            [W C P Q] = obj_func(cv_Xtr, cv_Ytr, lambda1_range(i), lambda2_range(ii), opts);
            opts.init=1;
            opts.P0=P;
            opts.Q0=Q;
            opts.C0=C;
            perform_mat{i}(ii) = perform_mat{i}(ii) + eval_func(cv_Yte, cv_Xte, W, C);
        end
    end
end

perform_mat=cell2mat(perform_mat)';
perform_mat = perform_mat./cv_fold;
fprintf(']\n')
    
[lambda1_idx  lambda2_idx] = find(perform_mat==max(max(perform_mat)));
best_lambda1=lambda1_range(lambda1_idx(end));
best_lambda2=lambda2_range(lambda2_idx(end));
end

