function [x, fval, exitflag, output]=SQP_matlab(f,confun,X0,tol)
% Set options with tolerance and display iterations
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter','StepTolerance', tol);

% Call fmincon
[x, fval, exitflag, output] = fmincon(f, X0, [], [], [], [], [], [], confun, options);
end