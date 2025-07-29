function iden_out = runge_kutta(ufunc,iden,x_input,h,t)
  


 %% 四阶龙格-库塔法
%     if size(x,1) == 1 %% if x is row vector
%         x = x';
%     end
    %%% (t):k, (t+h/2):k+1, (t+h):k+2
    k1 = ufunc(t, iden, x_input(:,1));
    k2 = ufunc(t+h/2, iden+k1.*h/2, x_input(:,2));
    k3 = ufunc(t+h/2, iden+k2.*h/2, x_input(:,2));
    k4 = ufunc(t+h, iden+k3.*h, x_input(:,3));
    iden_out = iden + h*(k1+2*k2+2*k3+k4)/6;
    
%     if size(x_out,1) == 1 %% if x_out is row vector
%         x_out = x_out';
%     end

end