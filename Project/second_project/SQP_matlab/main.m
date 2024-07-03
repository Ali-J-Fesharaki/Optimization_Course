clear all
clc
format short
tol_SQP=1e-6; % uncertaineity of Condugate gradient method

response = input(['if you want to solve cardinal problem insert "1" then press inter' ...
    ' \nelse if you want to define a new function insert "2" then press inter\n' ...
    '**pressing any other key will solve a sample quadratic problem **\n']);
if response==1
    X0 = [-1.8, 1.7, 1.9, -0.8, -0.8];
    [x, fval, exitflag, output] =SQP_matlab(@f_1,@confun_1,X0,tol_SQP);
    
elseif response==2
    X0=[3, 4]';
    [x, fval, exitflag, output] =SQP_matlab(@f_2,@confun_2,X0,tol_SQP);

end
% Display results
disp('Optimized variables:');
disp(x);
disp('Function value:');
disp(fval);
disp('Exit flag:');
disp(exitflag);
disp('Output details:');
disp(output);


% Objective function
function f = f_2(x)
    f = (x(1)^2 + x(2) - 11)^2 + (x(1) + x(2)^2 - 7)^2;
end

% Constraints function
function [c, ceq] = confun_2(x)
    c = zeros(2, 1);
    ceq = [];
    c(1) = -((x(1) + 2)^2 - x(2));
    c(2) = -(-4 * x(1) + 10 * x(2));
end

% Objective function
function f = f_1(x)
    f = exp(x(1) * x(2) * x(3) * x(4) * x(5)) - 0.5 * (x(1)^3 + x(2)^3 + 1)^2;
end

% Constraints function
function [c, ceq] = confun_1(x)
    ceq = zeros(3, 1);
    c= [];
    ceq(1) = x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 + x(5)^2 - 10;
    ceq(2) = x(2) * x(3) - 5 * x(4) * x(5);
    ceq(3) = x(1)^3 + x(2)^3 + 1;
end