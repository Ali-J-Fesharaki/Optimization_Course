clear all
clc
format long
tol_FR=1e-4; % uncertaineity of Condugate gradient method
tol_CF=1e-4; % uncertaineity of Condugate gradient method
tol_GS=1e-4; % uncertaineity of Golden Section method
N=300; % number of iterations in golden section method
response = input(['if you want to solve cardinal problem insert "1" then press inter' ...
    ' \nelse if you want to define a new function insert "2" then press inter\n' ...
    '**pressing any other key will solve a sample quadratic problem **\n']);
if response==1
    x0=[0, 0, 0]'
    n=size(x0,1);
    x=sym("x", [n 1]);
    f=(4-x(1))^2+(4-x(2))^2+45*(x(2)-x(1)^2)^2+45*(x(3)-x(2)^2)^2
    order=4;
elseif response==2
    x0=[0, 0]';
    n=size(x0,1);
    x=sym("x", [n 1]);
    f=(x(1)*x(2)-x(1)+1.5)^2+(x(1)*x(2)^2-x(1)+2.25)^2+(x(1)*x(2)^3-x(1)+2.625)^2
    order=6;
elseif response==3 
    x0(:,1)= input('input initial value of variables in the column form(like :[-1;3] ) \n')
    n=size(x0,1);
    x=sym("x", [n 1]);
    f = input('define new cost function like f=(1- x(1))^2 +(1- x(2))^2 ... \n')
    order=input('input the order of your cost function: \n')
end
[optimum_point,optimum_value] = conjugate_gradient(f,x,x0,tol_FR,tol_GS,N,order);

