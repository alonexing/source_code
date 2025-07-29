function x_hat_dot = FI_estimator(t,x_hat,x)
    
    global A center eta WBar X_min X_max;

    %% S的生成，DLearing、Exp_identify、FI_estimator_NEW三个文件要保持一致 
    % x_normalized = [(x(4:end) - X_min') ./ (X_max' - X_min')];
    x_normalized = [(x(4:end-1) - X_min') ./ (X_max' - X_min');x(end,:)];
    S = RBFNN(x_normalized,center,eta);   %% 进行归一化 
    % S = RBFNN(x(4:end),center,eta);  %% 不进行归一化 
    %% 
    % WBar(:,4:end) =WBar(:,4:end)*1;
    % WBar(:,1:3) = WBar(:,1:3);
    x_hat_dot = repmat(A,2,1).*(repmat(x(1:3,1),2,1) - x_hat) + WBar(:,1:end)'*S ;   
    
end