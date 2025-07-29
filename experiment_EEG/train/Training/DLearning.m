function iden_dot = DLearning(t,iden,x)
    
    global eta center X_min X_max;

    %%% 
    A = 4; Gamma = 0.35; sigma = 0.001; 

    %% S的生成，DLearing、Exp_identify、FI_estimator_NEW三个文件要保持一致 
    % x_normalized = [(x(4:end) - X_min') ./ (X_max' - X_min')];
    x_normalized = [(x(4:end-1) - X_min') ./ (X_max' - X_min');x(end,:)];
    S = RBFNN(x_normalized,center,eta);   %% 进行归一化 
    % S = RBFNN(x(4:end),center,eta);  %% 不进行归一化 
    %% 
    x_hat = iden(1,:)'; W_hat = iden(2:end,:);
    x_hat_dot = A*(x(1:3,1) - x_hat) + W_hat'*S ;
    W_hat_dot = (Gamma * (x(1:3,1) - x_hat) * S' - sigma * Gamma * W_hat')';
    iden_dot = [x_hat_dot';W_hat_dot];
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
end