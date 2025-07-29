function [c,f,s] = sys_pde(z,t,x,DuDx)

    global index1 index2 index3 index4 tc1 tc2 tc3 tc4 ;
        % 定义故障类型与时间阈值，变量个数需与testdata_gen保持一致
    fault_times = [tc1, tc2, tc3,tc4]; % 激活故障的时间示例
    fault_modes = [index1, index2, index3,index4];    % 对应的故障模式
    
    % 根据时间确定当前故障模式
    current_mode = 0; % 初始设定为正常模式
    for i = 1:length(fault_times)
        if t >= fault_times(i)
            current_mode = fault_modes(i);
        end
    end   
    switch current_mode
    %%% normal mode
        case 0
            betaT = 50; gamma = 4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);   
            phi = 0;
            
    %%% faulty mode #1 actuator fault  修改参数gamma（活化能）
        case 1  %% trained/test
            betaT = 50; gamma = 3.5; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);  
            phi = 0;
        case 11 %% match
            betaT = 50; gamma = 3.6; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);  
            phi = 0;
        case 111 %% match
            betaT = 50; gamma = 3.4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);  
            phi = 0;
            
    %%% faulty mode #2 (partial location)state fault  修改参数phi（状态故障）
        case 2  %% trained/test
            betaT = 50; gamma = 4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);
            phi = (heaviside(z-1)-heaviside(z-1.3))*(1*x);  
        case 22 %% match
            betaT = 50; gamma = 4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);
            phi = (heaviside(z-1)-heaviside(z-1.2))*(1*x);  
        case 222 %% match
            betaT = 50; gamma = 4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);
            phi = (heaviside(z-1.1)-heaviside(z-1.35))*(1*x);
            
    %%% faulty mode #3 parameter fault   修改参数betaT（反应热）
        case 3  %% trained/test
            betaT = 48; gamma = 4; betaU = 2;
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);    
            phi = 0;
        case 33 %% match
            betaT = 49; gamma = 4; betaU = 2; 
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);    
            phi = 0;
        case 333 %% match
            betaT = 48.5; gamma = 4; betaU = 2; 
            B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);    
            phi = 0;
            
    %%% mismatch fault   未知模式，修改参数B（执行器分布函数）
        case 4  %% trained/test
            betaT = 50; gamma = 4; betaU = 2;
            B = 1*sin(0.3*z)+1*sin(0.6*z)+1*sin(0.9*z);
            phi = 0;
         
    end
    
    c = 1;
    f = DuDx;
    U = 1.1+2*sin(5*t)-2*cos(5*t);
    s = betaT*(exp(-gamma/(1+x))-exp(-gamma))+betaU*(B*U'-x)+phi;   
    
end