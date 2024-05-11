clc
clear
format long
options = optimset;
x0=[1,-2,5]';
options = optimset(options,'Display', 'iter');
options = optimset(options,'FunValCheck', 'off');
[x,fval,exitflag,output] =fminsearch(@(x)(1-x(1))^2+(1-x(2))^2+50*(x(2)-x(1)^2)^2+50*(x(3)-x(2)^2)^2,x0,options)