close all; 
clear all; clc; close all; 
global steps Ts A center eta WBar X_min X_max;  

load E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\FDI_design_TRYnew.mat;
load E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\testdata_new.mat;

%% %%%%%%%%%%%%%%%%%% initial condition & parameters

rho_int = 0;      unknown_mode = 9;% 未知模式的编号
a = 0.1; A = [a;a;a]; %% parameters of threshold
%KT：求误差的滑动时间窗，KT_FI：防止误判的时间窗，threshold：未知模式的误差阈值，target_array：测试数据的模式顺序
hbar = 0.06;   Ts = 0.01;  num = 3;KT_FI = 23; threshold =1.5;target_array = [1, 2];
%数据的时间步设置
dt = 0.001; t = 0:dt:51.199; 
U = 1.1+2*sin(5*t)-2*cos(5*t); %% system input 
 Tk = Ts/(t(2)-t(1)); steps = 0.999*size(t,2)/Tk; KT = 1.8/Ts; ep = 2/Ts+KT;
Timek = t(1,1:Tk/2:end); xk_iden = test_red(1:Tk/2:end,:)'; NNinputk = [test_red(1:Tk/2:end,:), U(1,1:Tk/2:end)']';
% NNinputk = [test_red(1:Tk/2:end,:)]';  
% 按同样时间步采样的原始数据x(z,t)
xk_iden_raw = test_raw(1:Tk/2:end,:)';
XK_IDEN_raw = xk_iden_raw(:,2:2:end);
XK_IDEN_red = xk_iden(:,2:2:end);
%% %%%%%%%%%%%%%%%%%% FDI 
FI_state = repmat(xk_iden(1:3,1),2,1);
recognized_time = []; 
recognized_values = [];
current_state = NaN;
unknown_mode_flag = false;
for k = 1:steps  
    x_iden = [xk_iden(:,(2*k-1):(2*(k+1)-1)); NNinputk(:,(2*k-1):(2*(k+1)-1))]; 
    % 每一步的辨识器生成数据xbar
    FI_state = runge_kutta(@FI_estimator_NEW,FI_state,x_iden,Ts,0);
    % 重建回xbar(z,t)
    [model0_gen(:,k)] = FI_state(1:3,:)';
    [model1_gen(:,k)] = FI_state(4:6,:)';
    % [model2_gen(:,k)] = KLD_construct(FI_state(7:9,:)', num, s_eigvector);
    % [model3_gen(:,k)] = KLD_construct(FI_state(10:12,:)', num, s_eigvector);

    % 对(xbar(z,t) - x(z,t))^2,空间层面求和*dz，然后开根号
    error0_S(:,k) = sqrt(sum((XK_IDEN_red(:,k)-model0_gen(:,k)).^2, 1));
    error1_S(:,k) = sqrt(sum((XK_IDEN_red(:,k)-model1_gen(:,k)).^2, 1));
    % error0_S(:,k) = XK_IDEN_red(3,k)-model0_gen(2,k);
    % error1_S(:,k) = XK_IDEN_red(2,k)-model1_gen(2,k);
    % error2_S(:,k) = sqrt(sum((XK_IDEN_raw(:,k)-model2_gen(:,k)).^2, 1) * dz);
    % error3_S(:,k) = sqrt(sum((XK_IDEN_raw(:,k)-model3_gen(:,k)).^2, 1) * dz);

    % 基于滑动时间窗，对时间层面求平均
    if k > KT
        error0_T(:,k) = mean(error0_S(:,k-KT:k),2);
        error1_T(:,k) = mean(error1_S(:,k-KT:k),2);
        % error2_T(:,k)= mean(error2_S(:,k-KT:k),2);
        % error3_T(:,k)= mean(error3_S(:,k-KT:k),2);

    %%% decision making
    % FI_decision为每一步的识别结果，recognized_values为识别到的模式切换结果，recognized_time为识别到的模式切换时间
        all_error = [error0_T(:,k);error1_T(:,k)];
        [min_error,FI_decision(:,k)] = min(all_error);
        current_value = FI_decision(:, k); % 当前时间点的数据

        % 获取前 KT_FI 个点
        current_values = FI_decision(:, (k-KT_FI+1):(k)); 
        
        % 检查这 KT_FI 个时间点是否都等于 current_value
        if all(all(current_values == current_values(:, 1)))
            new_state = current_value(1); % 用第一行的值作为模式
            
            if min_error < threshold  % 识别到已知模式
                if (isnan(current_state) || new_state ~= current_state)
                    % 如果状态变化，记录当前的 k（乘以 Ts）和相应的识别值
                    recognized_time(end + 1) = k * Ts; % 记录识别时间
                    recognized_values(end + 1) = new_state; % 记录当前识别的模式
                    % 更新当前状态
                    current_state = new_state;
                    unknown_mode_flag = false; % 重置未知模式标志
                end
            elseif min_error >= threshold && ~unknown_mode_flag % 识别到未知模式
                % 记录未知模式
                recognized_time(end + 1) = k * Ts; % 记录识别时间
                recognized_values(end + 1) = unknown_mode; % 记录未知模式的识别值
                
                % 更新当前状态为未知模式
                current_state = unknown_mode;
                unknown_mode_flag = true; % 设置未知模式标志
            end
        else
            % 如果不一致，重置当前状态
            current_state = NaN;

        end
    end
    % if k > KT_FI

    % end
    k
end
if isequal(recognized_values(1), target_array)  % 如果识别出的模式与测试数据模式相同，则认为成功
    disp('识别成功');
else
    disp('识别失败');
end
%% %%%%%%%%%%%%%%% PLOTing
% %分别绘制不同的辨识器对应的测试数据动力学误差
% figure, box on,
% plot(error0_T(:,:)');
% 
% figure, box on,
% plot(error1_T(:,:)');
% 
% %统合绘制不同的辨识器对应的测试数据动力学误差
% figure;
% 
% % 正常模式辨识器
% plot(error0_T', 'r-', 'DisplayName', 'Normal'); 
% hold on; % 保持当前图形，以便继续添加其他变量
% 
% % 故障1模式辨识器
% plot(error1_T', 'g--', 'DisplayName', 'Fault1'); 
% 
% % 添加图例
% legend('show');
% 
% % 添加标题和标签
% title('dynamic error');
% xlabel('step');
% ylabel('value');
% 
% % 显示网格
% grid on;
% 假设变量为 var1 和 var2
% 它们都是大小为5994的向量

N = 5110;
num_blocks = 64;
block_size = floor(N / num_blocks);  % 若不整除，可调整

% 示例随机变量（实际用你的变量替换）
% var1 = rand(1, N);
% var2 = rand(1, N);

% 计算差值
diff_vars = error0_S- error1_S;

% 初始化结果
patterns = zeros(1, num_blocks);

for i = 1:num_blocks
    start_idx = (i - 1) * block_size + 1;
    if i < num_blocks
        end_idx = i * block_size;
    else
        end_idx = N;  % 最后一个块包括剩余部分
    end
    
    block_diff = diff_vars(start_idx:end_idx);
    count_positive = sum(block_diff > 0);
    count_negative = sum(block_diff < 0);
    count_total = length(block_diff);
    
    % 阈值：大于一半
    if i <= 32
        % 前6块
        if count_positive > count_total / 2
            patterns(i) = 1;  % 识别为模式1
        else
            patterns(i) = 0;  % 识别为模式0
        end
    else
        % 后6块
        if count_negative < count_total / 2
            patterns(i) = 1;  % 识别为模式1
        else
            patterns(i) = 0;  % 识别为模式0
        end
    end
end

% 输出结果
disp('块的识别结果（0或1）：');
disp(patterns);
accuracy1 = sum(patterns == y_test') / numel(patterns);
disp(accuracy1);