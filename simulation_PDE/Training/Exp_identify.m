function x_hat_dot = Exp_identify(t,x_hat,x)
    
    global eta center Wb X_min X_max;

    A = 4;

    %% S的生成，DLearing、Exp_identify、FI_estimator_NEW三个文件要保持一致 
    x_normalized = [(x(4:end-1) - X_min') ./ (X_max' - X_min');x(end,:)];
    S = RBFNN(x_normalized,center,eta);   %% 进行归一化 
    
    % S = RBFNN(x(4:end),center,eta);  %% 不进行归一化 
    %% 
    x_hat_dot = A*(x(1:3,1) - x_hat) + Wb'*S;  

end