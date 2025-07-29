
clear all; clc; %close all;

global index1 index2 index3 index4 tc1 tc2 tc3 tc4;
load D:\PDE_system\traindata_new.mat

%% %%%%%%%%%%%%% 生成测试数据
%%% 为测试数据设置参数，参数保存后会用于MAIN_FDI_NEW中的识别，z, num, s_eigvector均来自训练数据
%%% n代表数字，tcn为模式切换时间，indexn为模式对应序号，需要有n个模式就设置n个两种变量
%%% sys_pde_test里的两种变量个数需与其保持一致
%%%% fault occurrence time
tc1 = 20; tc2 = 40.001;tc3 = 60.002; tc4 = 80.003;
%%%% fault type
index1 = 111; index2 = 222; index3 = 333;index4 = 4;
num = 3;
% t: time dimension; x: system state
dt = 0.001; t = 0:dt:100.004; 
U = 1.1+2*sin(5*t)-2*cos(5*t); %% system input

% 生成用于测试的原始、降维数据
sol = pdepe(0,@sys_pde_test,@sys_ic,@sys_bc,z,t);
x = sol(:,:,1); %% system state  
test_raw = x;
data = test_raw;

[red_data] = KLD_reduce(data, num, s_eigvector);
test_red = red_data;



save D:\PDE_system\testdata_new.mat test_raw test_red s_eigvector ...
     index1 index2 index3 tc1 tc2 tc3 U t z;



%% %%%%%%%%%%%% PLOTing
% %%%%%
figure, box on,
dz = z(2)-z(1); dt = t(2)-t(1);

zax = z(1:18:end);
tax = t(1:400:end);
dataax = test_raw(1:400:end,1:18:end);
surf(zax,tax,dataax) ;
% xlim([0,15]);
xlabel({'z'},'Interpreter','Latex')
ylabel({'t'},'Interpreter','Latex')
zlabel({'x(z,t)'},'Interpreter','Latex')
title('data')
hold on;


