function [pl,ql,pr,qr] = sys_bc(zl,xl,zr,xr,t)
pl = xl;
ql = 0;
pr = xr;
qr = 0; 