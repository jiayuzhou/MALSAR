%% FUNCTION Logistic_TGL
%  L21 Joint Feature Learning with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + opts.rho_L2 * \|W\|_2^2 + rho1 * \|W\|_{2,1} }
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: L2,1-norm group Lasso parameter.
%   OPTIONAL
%      opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
%   W: model: d * t
%   C: model: 1 * t
%   funcVal: function value vector.
%
%% Code starts here
function [W, C, P, Q, L, F] = my_Logistic_rMTL(X, Y, rho1, rho2, opts)

if nargin <4
    error('\n Inputs: X, Y, rho1, should be specified!\n');
end
X = multi_transpose(X);

if nargin <5
    opts = [];
end



% initialize options.
%opts=init_opts(opts);

task_num  = length (X);
dimension = size(X{1}, 1);
L = [];
F = [];

%initialize a starting point
C0_prep = zeros(1, task_num);
for t_idx = 1: task_num
    m1 = nnz(Y{t_idx} == 1);
    m2 = nnz(Y{t_idx} == -1);
    if ( m1==0 || m2==0 )
        C0_prep(t_idx) = 0;
    else
        C0_prep(t_idx) = log(m1/m2);
    end
end

if opts.init==2
    P0 = zeros(dimension, task_num);
    Q0 = zeros(dimension, task_num);
    C0 = zeros(1, task_num);
elseif opts.init== 0
    P0 = randn(dimension, task_num);
    Q0 = randn(dimension, task_num);
    C0 = C0_prep;
elseif opts.init== 1
    if isfield(opts,'P0')
        P0=opts.P0;
        if (nnz(size(P0)-[dimension, task_num]))
            error('\n Check the input .P0');
        end
    else
        error('\n check opt.init');
    end
    
     if isfield(opts,'Q0')
        Q0=opts.Q0;
        if (nnz(size(Q0)-[dimension, task_num]))
            error('\n Check the input .Q0');
        end
    else
        error('\n check opt.init');
    end   
    
    if isfield(opts,'C0')
        C0=opts.C0;
    else
        error('\n check opt.init');
    end
end




Pz= P0;
Qz= Q0;
Cz= C0;
Pz_old = P0;
Qz_old = Q0;
Cz_old = C0;

t = 1;
t_old = 0;
iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ps = (1 + alpha) * Pz - alpha * Pz_old;
    Qs = (1 + alpha) * Qz - alpha * Qz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    
    % compute function value and gradients of the search point
    [gWs, gCs, Fs ]  = gradVal_eval(Ps+Qs, Cs);
    
    % the Armijo Goldstein line search scheme
    while true
        Pzp = FGLasso_projection(Ps - gWs/gamma, rho1 / gamma);
        Qzp = FGLasso_projection(Qs' - gWs'/gamma, rho2 / gamma)';

        Czp = Cs - gCs/gamma;
        Fzp = funVal_eval(Pzp+Qzp, Czp);
        
        delta_Pzp = Pzp - Ps;
        delta_Qzp = Qzp - Qs;
        delta_Czp = Czp - Cs;
        nm_delta_Pzp=norm(delta_Pzp, 'fro')^2;
        nm_delta_Qzp=norm(delta_Qzp, 'fro')^2;
        nm_delta_Czp=norm(delta_Czp, 'fro')^2;

        r_sum = (nm_delta_Pzp+nm_delta_Qzp+nm_delta_Czp)/2;
        
%         Fzp_gamma = Fs + sum(sum((delta_Pzp+delta_Qzp).* gWs))...
%             + sum(sum(delta_Czp .* gCs))...
%             + gamma/2 * norm(delta_Pzp + delta_Qzp, 'fro')^2 ...
%             + gamma/2 * nm_delta_Czp;
                    
        Fzp_gamma = Fs + sum(sum((delta_Pzp+delta_Qzp).* gWs))...
            + sum(sum(delta_Czp .* gCs))...
            + gamma/2 * (nm_delta_Pzp +nm_delta_Qzp) ...
            + gamma/2 * nm_delta_Czp;
                    
                    
        if (r_sum <=1e-20)
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Pz_old = Pz;
    Qz_old = Qz;
    Cz_old = Cz;
    Pz = Pzp;
    Qz = Qzp;
    Cz = Czp;
    
    L = cat(1, L, Fzp + nonsmooth_eval(Pzp, rho1) + nonsmooth_eval(Qzp', rho2));
    F = cat(1, F, Fzp);

    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( L(end) - L(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( L(end) - L(end-1) ) <=...
                        opts.tol* L(end-1))
                    break;
                end
            end
        case 2
            if ( L(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

P=Pzp;
Q=Qzp;
W = Pzp+Qzp;
C = Czp;

% private functions

    function [Wp] = FGLasso_projection (W, lambda )
        % solve it in row wise (L_{2,1} is row coupled).
        % for each row we need to solve the proximal opterator
        % argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        Wp = repmat(max(0, 1 - lambda./sqrt(sum(W.^2,2))),1,size(W,2)).*W;

    end


    function [grad_W, grad_C, funcVal] = gradVal_eval(W, C)
        grad_W = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);
        for i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i});
        end
        % here when computing function value we do not include
        % l1 norm.
        funcVal = sum(lossValVect);
    end

    function [funcVal] = funVal_eval (W, C)
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
        end
        % here when computing function value we do not include
        % l1 norm.
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho)
        non_smooth_value = sum(sqrt(sum(W.^2, 2)))*rho;        
    end

    function [Xnorm] = L1infnorm(X)
    % ||X||_{1,2} = sum_i||X^i||_inf
    Xnorm = sum(max(abs(X),[],2));
    end

    function [Xnorm] = L11norm(X)
    % ||X||_tr = sum_i\sigma_i
    Xnorm = sum(sum(abs(X)));
    end

    function [X] = proximalL1infnorm(D, tau)
        % min_X 0.5*||X - D||_F^2 + tau*||X||_{1,inf}
        % where ||X||_{1,inf} = sum_i||X^i||_inf, where X^i denotes the i-th row of X
        
        % X = D; n = size(D,2);
        % for ii = 1:size(D,1)
        %     [mu,~,~] = prf_lb(D(ii,:)', n, tau);
        %     X(ii,:) = D(ii,:) - mu';
        
        % end
        
        [m,n]=size(D);
        [mu,~,~]=prf_lbm(D,m,n,tau);
        X = D - mu;     
    end


    function [X] = proximalL11norm(D, tau)
        % min_X 0.5*||X - D||_F^2 + tau*||X||_{1,1}
        % where ||X||_{1,1} = sum_ij|X_ij|, where X_ij denotes the (i,j)-th entry of X
        X = sign(D).*max(0,abs(D)-tau);
    end
end

function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y)
%gradient and logistic evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_c = sum(b);
grad_w = x * b;
end

function [ funcVal ] = unit_funcVal_eval( w, c, x, y)
%function value evaluation for each task 
m = length(y);
weight = ones(m, 1)/m;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end
