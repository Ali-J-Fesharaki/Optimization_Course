clc;
clear;
response = input(['if you want to solve cardinal problem insert "1" then press inter' ...
    ' \nelse if you want to define a new function insert "2" then press inter\n' ...
    '**pressing any other key will solve a sample quadratic problem **\n']);
if response==1
    x0=[0, 0, 0]';
    n=size(x0,1);
    x=sym("x", [n 1]);
    f=@(x)(4-x(1))^2+(4-x(2))^2+45*(x(2)-x(1)^2)^2+45*(x(3)-x(2)^2)^2
    order=4;
elseif response==2
    x0=[0, 0]';
    n=size(x0,1);
    x=sym("x", [n 1]);
    f=@(x)(x(1)*x(2)-x(1)+1.5)^2+(x(1)*x(2)^2-x(1)+2.25)^2+(x(1)*x(2)^3-x(1)+2.625)^2
    order=6;
end
% Create options structure with desired settings
options = optimset;
options = optimset(options, 'Display', 'iter'); % Show iteration output
options = optimset(options, 'TolX', 1e-4);     % Tolerance on the change in variables

% Perform the optimization
[x, fval, exitflag, output] = fminsearch(f, x0, options);

% Display the results
disp('Optimal point:');
disp(x);
disp('Function value at optimal point:');
disp(fval);
disp('Exit flag:');
disp(exitflag);
disp('Output structure:');
disp(output);
