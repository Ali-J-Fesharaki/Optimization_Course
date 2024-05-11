clc
clear
options = optimoptions('fminunc');
options = optimoptions(options,'Display', 'iter');
options = optimoptions(options,'FunValCheck', 'off');
options = optimoptions(options,'Algorithm', 'quasi-newton');
options = optimoptions(options,'Diagnostics', 'on');
x0=[1,-2,5]';
[x,fval,exitflag,output,grad,hessian] = ...
fminunc(@(v)(1-v(1))^2+(1-v(2))^2+60*(v(2)-v(1)^2)^2+60*(v(3)-v(2)^2)^2,x0,options)
