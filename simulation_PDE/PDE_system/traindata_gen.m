
clear all; clc; %close all;
global index X_max X_min tc;

%% %%%%%%%%%%%%% 生成训练数据
%%% 为训练数据设置参数，参数保存后会用于MAIN_training_NEW中的训练
dz = 0.005; dt = 0.001; 
num = 3; %主成分数量
num_faults = 4;%训练数据的故障数目
z = 0:dz:pi; t = 0:dt:100; tc = 0;
index_h = [0,1,2,3]; % 训练数据的故障类型;
U = 1.1+2*sin(5*t)-2*cos(5*t); %% system input

faults = struct(); % 用于存储故障系统数据
total_data = zeros(num_faults*size(t,2), size(z,2));
for h = 1:num_faults
    index = index_h(h);
    sol = pdepe(0,@sys_pde_train,@sys_ic,@sys_bc,z,t);
    x = sol(:,:,1); %% system state
    train_raw = x;
    faults.(sprintf('fault%d', h-1)).x = train_raw; % 储存当前循环内生成的单一故障系统数据
    total_data((h-1)*size(t,2) + 1:h*size(t,2), :) = train_raw;% 拼接并储存所有故障系统数据
end
%%% original system
% z: spatial dimension; t: time dimension; x: system state

data = total_data;
[s_eigvector, evr] = KLDfeatures_cal(data);
% 归一化所需变量
[total_red] = KLD_reduce(data, num, s_eigvector);% 降维后的所有故障系统数据
X_min = min(total_red);
X_max = max(total_red);
for h = 1:num_faults
    [red_data] = KLD_reduce(faults.(sprintf('fault%d', h-1)).x, num, s_eigvector);
    train_red = red_data;
    faults.(sprintf('fault%d', h-1)).xs = train_red; % 储存降维后的单一故障系统数据
end


save D:\PDE_system\traindata_new.mat faults s_eigvector evr num_faults...
     index U t z X_min X_max;

%% %%%%%%%%%%%% PLOTing
% %%%%%
figure, box on,
dz = z(2)-z(1); dt = t(2)-t(1);
t = 0:dt:400.003;
zax = z(1:18:end);
tax = t(1:4000:end);
dataax = total_data(1:4000:end,1:18:end);
surf(zax,tax,dataax) ;
% xlim([0,15]);
xlabel({'z'},'Interpreter','Latex')
ylabel({'t'},'Interpreter','Latex')
zlabel({'x(z,t)'},'Interpreter','Latex')
title('data')
hold on;


