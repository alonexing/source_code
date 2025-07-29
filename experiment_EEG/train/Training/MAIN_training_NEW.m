% clear all; clc
close all;
global N eta center steps Ts Wb X_min X_max; 

load E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\traindata_new.mat;

%% %%%%%%%%%%%%%%%%%%%%% Generate Nueral Network %%%%%%%%%%%%%%%%%%%%%%%%

dt = 0.001; t = 0:dt:74.999; 
eta = 1;
% b = 1;   % RBF function width: b*eta
% 3 input
c1 = -5:eta:5; c2 = -5:eta:5; c3 = -5:eta:5; c4 = -5:eta:5;
N = length(c1)*length(c2)*length(c3)*length(c4);
index = 1;
%Generate center matrix, whose ith column is the coordinate of ith neural
center = zeros(4,N);
for N_1 = 1:length(c1)
    for N_2 = 1:length(c2)
        for N_3 = 1:length(c3)
            for N_4 = 1:length(c4)
                center(:,length(c4)*(length(c3)*(length(c2)*(N_1-1) + N_2-1)+N_3-1)+N_4) = [c1(N_1); c2(N_2); c3(N_3); c4(N_4)];
            end
        end
    end
end

%Generate center matrix, whose ith column is the coordinate of ith neural
% center = zeros(3,N);
% index = 1; % 计数器
% for N_1 = 1:length(c1)
%     for N_2 = 1:length(c2)
%         for N_3 = 1:length(c3)
%             center(:, index) = [c1(N_1); c2(N_2); c3(N_3)];
%             index = index + 1;
%         end
%     end
% end

%%%%%%%%%%%%%%%% train
Ts = 0.01; Tk = Ts/(0.001); steps = size(t,2)/Tk; Timek = t(1,1:Tk/2:end); ktime = steps/1.01; 

for h = 0:num_faults-1
    % 逐模式读取训练数据，每次循环代表一个模式的训练
    train_red_h = faults.(sprintf('fault%d', h)).xs;

    %%%% identified state
    xk_iden = train_red_h(1:Tk/2:end,:)';
    NNinputk = [train_red_h(1:Tk/2:end,:), U(1,1:Tk/2:end)']';
    % NNinputk = [train_red_h(1:Tk/2:end,:)]';
%%%%%%%%%%%%%%%%%% DL identification (for Weight_bar)
    
    i = 1; xh = xk_iden(:,1); W = zeros(N,3); iden_hat = [xh'; W];
    for k = 1:ktime 

        xhk(:,k) = iden_hat(1,1:3); 
        x_iden = [xk_iden(:,(2*k-1):(2*(k+1)-1)); NNinputk(:,(2*k-1):(2*(k+1)-1))]; 
        iden_hat = runge_kutta(@DLearning,iden_hat,x_iden,Ts,0);   
        W_hat(:,:,k) = iden_hat(2:end,:);
        if k >= ktime-1000
            W_conv(:,:,i) = iden_hat(2:end,:); 
            i = i+1;
        end 
        k
    end
    Wb = mean(W_conv,3);  
    
%%%%%%%%%%%%%%%%%% experience testing (for Epsilon)
 
    KT = 1/Ts; ep = 10/Ts; iden_bar = xk_iden(:,1); i = 1;
    for k = 1:ktime/2   
        
        xhbk(:,k) = iden_bar; 
        x_iden = [xk_iden(:,(2*k-1):(2*(k+1)-1)); NNinputk(:,(2*k-1):(2*(k+1)-1))]; 
        iden_bar = runge_kutta(@Exp_identify,iden_bar,x_iden,Ts,0);

        emk(:,k) = iden_bar - xk_iden(:,2*k+1);
        if k >= ep
            emi(:,i) = mean(abs(emk(:,k-KT:k)),2);
            i = i+1;
        end       
        k
    end

%%%%%%%%%%%%%%%%%%% plot
    Weight_bar{h+1} = Wb; 
    % Weight_conv{h+1} = W_conv; EMK_total{h+1} = emk;% 变量备用于后续画图
    % Weight_hat{h+1} = W_hat;EMI_total{h+1} = emi;
    XHK_all{h+1}= xhk;
    XKIDEN_all{h+1}= xk_iden;
    Kp = size(xhk,2);
    figure, box on, hold on,
    plot(Timek(1,1:2:2*Kp),xhk(3,1:Kp));
    plot(Timek(1,1:2:2*Kp),xk_iden(3,1:2:2*Kp));
    title('xhk vs. xk_iden');
    hold off
    % % % 
    % %%%%% experience testing
    Kp = size(emk,2);
    figure, box on,
    plot(Timek(1,1:2:2*Kp),emk(:,1:Kp)');
    title('emk');
    % What的L2范数，多维变一维
    W_hat_L2norm = squeeze(vecnorm(W_hat, 2, 2));
    figure, box on,
    plot(W_hat_L2norm(1:200,1:5:end)');
    title('W_hat_L2norm');
end

% 用于存储训练得到的变量为结构体  % 变量备用于后续画图
% Weight_conv0 = Weight_conv{1} ;EMK_total0 = EMK_total{1};Weight_hat0 = Weight_hat{1};
% Weight_conv1 = Weight_conv{2} ;EMK_total1 = EMK_total{2};Weight_hat1 = Weight_hat{2};
% Weight_conv2 = Weight_conv{3} ;EMK_total2 = EMK_total{3};Weight_hat2 = Weight_hat{3};
% Weight_conv3 = Weight_conv{4} ;EMK_total3 = EMK_total{4};Weight_hat3 = Weight_hat{4};
% W_hat_L2norm_0 = squeeze(vecnorm(Weight_hat0, 2, 2));
% W_hat_L2norm_1 = squeeze(vecnorm(Weight_hat1, 2, 2));
% W_hat_L2norm_2 = squeeze(vecnorm(Weight_hat2, 2, 2));
% W_hat_L2norm_3 = squeeze(vecnorm(Weight_hat3, 2, 2));

WBar = [Weight_bar{1}, Weight_bar{2}];

% 用于保存mat文件
save E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\FDI_design_TRYnew.mat Weight_bar WBar center eta X_min U X_max;

