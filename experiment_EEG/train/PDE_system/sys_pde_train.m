function [c,f,s] = sys_pde(z,t,x,DuDx)

    global index tc;
    
    if t >= tc    
        switch index
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
    else
        betaT = 50; gamma = 4; betaU = 2;
        B = 1.5*sin(1*z)+1.8*sin(2*z)+2*sin(3*z);  
        phi = 0;
    end
    
    c = 1;
    f = DuDx;
    U = 1.1+2*sin(5*t)-2*cos(5*t);
    s = betaT*(exp(-gamma/(1+x))-exp(-gamma))+betaU*(B*U'-x)+phi;   
    
end