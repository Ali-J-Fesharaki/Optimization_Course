clear all
clc
close all
format long
tol_powell=1e-8; % uncertaineity of Golden Section method
tol_GS=1e-4; % uncertaineity of Golden Section method
N=200; % number of iterations in golden section method
response = input(['if you want to solve cardinal problem insert "1" then press inter' ...
    ' \nelse if you want to define a new function insert "2" then press inter\n' ...
    '**pressing any other key will solve a sample codratic problem **\n']);
if response==1
    x0=[1, -2, 5]';
    n=size(x0,1);
    x=sym("x", [n 1]);
    f=(1-x(1))^2+(1-x(2))^2+50*(x(2)-x(1)^2)^2+50*(x(3)-x(2)^2)^2 
    order=4;
elseif response==2 
    x0(:,1)= input('input initial value of variables in the column form(like :[-1;3] ) \n')
    n=size(x0,1);
    x=sym("x", [n 1]);
    f = input('define new cost function like f=(1- x(1))^2 +(1- x(2))^2 ... \n')
    order=input('input the order of your cost function: \n')
else % a quadratic problem (for testing code)
    x0=[0 ,0 ]';
    n=size(x0,1);
    x=sym("x", [n 1]);
    f= x(1)-x(2)+ 2*x(1)^2 + 2*x(1)*x(2) + x(2)^2;
    order=2;
end

[optimum_point, optimum_value,powell_iteration] = Powell(f,x,x0,order,tol_powell,tol_GS,N)


