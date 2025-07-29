
clear all; clc; %close all;
load E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\s_4.mat;
global X_max X_min;
% 读取.mat文件

% 提取x变量
X = ssn.x;  % 假设结构体名为ssn，变量名为x
% 参数设置
Fs = 1000; % 采样频率，单位：Hz
channels=30;
% 设计带通滤波器（1-30Hz）
d = designfilt('bandpassiir', ...
               'FilterOrder', 4, ...
               'HalfPowerFrequency1',0.1, ...
               'HalfPowerFrequency2', 4, ...
               'SampleRate', Fs);
% 对每个通道进行滤波
dt = 0.001; t = 0:dt:74.999; 
U = 1.1+2*sin(5*t)-2*cos(5*t); %% system input
% 提取第一个模式（前5000行）
NUM = 5000*15;
NUM1 = 5000*3;
NUM_CHANEL_TRAIN = 15;
NUM_CHANEL_TEST = NUM_CHANEL_TRAIN+1;
pattern1 = X(1:5000, :, 1:NUM_CHANEL_TRAIN);  % 大小为 5000 x 30 x 18
pattern1 = permute(pattern1, [1, 3, 2]);

% 再将数组reshape为 (5000*12, 30)
pattern1 = reshape(pattern1, [NUM , 30]);
% 提取第二个模式（第10001到15000行）
pattern2 = X(10001:15000, :, 1:NUM_CHANEL_TRAIN);  % 大小为 5000 x 30 x 18
pattern2 = permute(pattern2, [1, 3, 2]);

% 再将数组reshape为 (5000*12, 30)
pattern2 = reshape(pattern2, [NUM, 30]);
% 也可以将其reshape为（5000, 30*18），方便拼接
pattern1_reshaped_train = reshape(pattern1, [NUM , 30]);
pattern2_reshaped_train = reshape(pattern2, [NUM , 30]);
for ch = 1:channels
    pattern1_reshaped_train(:, ch) = filtfilt(d, pattern1_reshaped_train(:, ch));
end
for ch = 1:channels
    pattern2_reshaped_train(:, ch) = filtfilt(d, pattern2_reshaped_train(:, ch));
end
% 如果需要将两个模式合成一个大矩阵（比如用于训练）
combined_data_train = [pattern1_reshaped_train; pattern2_reshaped_train];  % 10000 x (30*18)
for ch = 1:channels
    combined_data_train(:, ch) = filtfilt(d, combined_data_train(:, ch));
end
pattern1 = X(1:5000, :, NUM_CHANEL_TEST:18);  % 大小为 5000 x 30 x 18
pattern1 = permute(pattern1, [1, 3, 2]);

% 再将数组reshape为 (5000*12, 30)
pattern1 = reshape(pattern1, [NUM1 , 30]);
% 提取第二个模式（第10001到15000行）
pattern2 = X(10001:15000, :, NUM_CHANEL_TEST:18);  % 大小为 5000 x 30 x 18
pattern2 = permute(pattern2, [1, 3, 2]);

% 再将数组reshape为 (5000*12, 30)
pattern2 = reshape(pattern2, [NUM1, 30]);
% 也可以将其reshape为（5000, 30*18），方便拼接
pattern1_reshaped = reshape(pattern1, [NUM1, 30]);
pattern2_reshaped = reshape(pattern2, [NUM1, 30]);
for ch = 1:channels
    pattern1_reshaped(:, ch) = filtfilt(d, pattern1_reshaped(:, ch));
end
for ch = 1:channels
    pattern2_reshaped(:, ch) = filtfilt(d, pattern2_reshaped(:, ch));
end
% 如果需要将两个模式合成一个大矩阵（比如用于训练）
combined_data_test = [pattern1_reshaped; pattern2_reshaped];  % 10000 x (30*18)
% combined_data_test = [pattern1_reshaped];  % 10000 x (30*18)
for ch = 1:channels
    combined_data_test(:, ch) = filtfilt(d, combined_data_test(:, ch));
end

test_raw = combined_data_test;
% 你可以根据需要保存或使用
%% %%%%%%%%%%%%% 生成训练数据
%%% 为训练数据设置参数，参数保存后会用于MAIN_training_NEW中的训练
dz = 0.005; dt = 0.001; 
num = 3; %主成分数量
num_faults = 2;%训练数据的故障数目
z = 0:dz:pi; t = 0:dt:100; tc = 0;
index_h = [0,1,2,3]; % 训练数据的故障类型;

faults = struct(); % 用于存储故障系统数据

data = combined_data_train;
[s_eigvector, evr] = KLDfeatures_cal(data);
% 归一化所需变量
[total_red] = KLD_reduce(data, num, s_eigvector);% 降维后的所有故障系统数据
[test_red] = KLD_reduce(test_raw, num, s_eigvector);
X_min = min(total_red);
X_max = max(total_red);
faults.(sprintf('fault0')).x = pattern1_reshaped_train; % 储存当前循环内生成的单一故障系统数据
faults.(sprintf('fault1')).x = pattern2_reshaped_train; % 储存当前循环内生成的单一故障系统数据
[TRAIN1_red] = KLD_reduce(pattern1_reshaped_train, num, s_eigvector);
[TRAIN2_red] = KLD_reduce(pattern2_reshaped_train, num, s_eigvector);
for h = 1:num_faults
    [red_data] = KLD_reduce(faults.(sprintf('fault%d', h-1)).x, num, s_eigvector);
    train_red = red_data;
    if h==1
    faults.(sprintf('fault%d', h-1)).xs = train_red; % 储存降维后的单一故障系统数据
    elseif h ==2
    faults.(sprintf('fault%d', h-1)).xs = train_red; % 储存降维后的单一故障系统数据
    end
end


save E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\traindata_new.mat faults s_eigvector evr num_faults...
     U X_min X_max;
save E:\tingjielunwenshiyan\EEG\EEG-Deformer-main\EEG-Deformer-main\data_processed\code\testdata_new.mat test_raw test_red s_eigvector U;
%% %%%%%%%%%%%% PLOTing
% %%%%%
% figure, box on,
% dz = z(2)-z(1); dt = t(2)-t(1);
% t = 0:dt:400.003;
% zax = z(1:18:end);
% tax = t(1:4000:end);
% dataax = total_data(1:4000:end,1:18:end);
% surf(zax,tax,dataax) ;
% % xlim([0,15]);
% xlabel({'z'},'Interpreter','Latex')
% ylabel({'t'},'Interpreter','Latex')
% zlabel({'x(z,t)'},'Interpreter','Latex')
% title('data')
% hold on;


