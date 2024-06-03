function [x,fval,exitflag,output] = Matlab_simplex(x0)
%% Start with the default options
options = optimset;
%% Modify options setting
options = optimset(options,'Display', 'iter');
options = optimset(options,'FunValCheck', 'off');
[x,fval,exitflag,output] = ...
fminsearch(@(v)(1-v(1))^2+(1-v(2))^2+50*(v(2)-v(1)^2)^2+50*(v(3)-v(2)^2)^2,x0,options);
