function [optimum_point,optimum_value] = BFGS (f,x,x0,tol_BFGS,tol_GS,N,order)
syms alf
format short
g_fob=gradient(f); % gradient of cost function
c(:,1) = subs(g_fob,x,x0);  % evaluate gradient at initial point
X(:,1)=x0;
func_eval=1;
n=size(x0,1);

if norm(c) < tol_BFGS
      optimum_point = x0;    % optimum point
      optimum_value = subs(f,x,x_opt); % cost function value
else 

beta(:,:,1)=eye(n);
k=1;
while vpa(norm(c(:,k)),2) > tol_BFGS && k < 3
    fprintf('\n*****  step %i:  *****\n',k)
    % solving this eqs: B(:,:,k)*d(:,k) = -c(:,k);
    v=sym("v", [n 1]);
    eq = beta(:,:,k)*v==-c(:,k);
    v = solve(eq, v);
    V=v;
    d(1,k)=V.v1;
    d(2,k)=V.v2;
    if n>2
        d(3,k)=V.v3;
    end

    if norm(d(:,k))>1
        d(:,k) = d(:,k)/norm(d(:,k));
    end

    if order>2
        [alfa_final,FE_GS] = golden_search(f,x,X(:,k),d(:,k),tol_GS,N);
        alfa=vpa(alfa_final,12)
        GS_iteration(k,:)=FE_GS;
    else
        alfa=vpa(solve(diff(subs(f,x,X(:,k)+alf*d(:,k)),alf)==0,alf),12)
        GS_iteration(k,:)=0;
    end
      X(:,k+1)=X(:,k) + alfa*d(:,k);
      c(:,k+1) = subs(g_fob,x,X(:,k+1));
      func_eval=func_eval+1; %function evaluaion for gradient
      p(:,k) = X(:,k+1)- X(:,k);
      q(:,k) = c(:,k+1)- c(:,k);

      if mod(k,n)==0
         beta(:,:,k)=eye(n);
      end
      D=(q(:,k)*q(:,k)')/(q(:,k)'*p(:,k));
      E=(c(:,k)*c(:,k)')/(c(:,k)'*d(:,k));
      beta(:,:,k+1) = beta(:,:,k) + D + E;
      
      fprintf('\nnorm of gradient is: %f     \n\n\n',norm(c(:,k+1)))
      cd=c(:,k+1)'*d(:,k);
      k=k+1;
end
    fprintf('\n ****************** final step is:%i ****************** \n ',k-1)
    fprintf('\nsum of the BFGS function evaluation is: %i \n',func_eval)
    fprintf('\nsum of the Golden Section evaluation is: %i \n',sum(GS_iteration))
    optimum_point = X(:,k) ;  
    optimum_value = vpa(subs(f,x,optimum_point),6);
end


%% saving Results
GS_iteration(k,:)=0;
x = vpa(X,5)';
BFGS_iteration=(0:k-1)';

    ResultsTable = table(BFGS_iteration, GS_iteration);
    for i = 1:n
        ResultsTable.(sprintf('x%d', i)) = double(x(:, i));
    end

    % Save the table to an Excel file without headers
    writetable(ResultsTable, 'BFGS_Results.xlsx')

end