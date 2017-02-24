function acc = eval_MTL_accuracy (Y, X, W, C)
%calculate accuracy for classification 

    task_num = length(X);    
    total_sample = 0;
    acc_list = zeros(task_num, 1);

    %calculate corrested classified numbers for every dataset
    for t = 1: task_num    
        acc_list(t) = nnz(sign(X{t} * W(:, t) + C(t)) == Y{t});
        total_sample = total_sample + length(Y{t});
    end 
    
    %calculate accuracy
    acc = sum(acc_list)/total_sample;
end
