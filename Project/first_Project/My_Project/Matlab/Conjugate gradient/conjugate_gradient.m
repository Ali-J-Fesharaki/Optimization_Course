function [optimum_point,optimum_value,k] = conjugate_gradient (f,x,x0,tol_FR,tol_GS,N,order)
syms alf
k=1;
format long
n=size(x0,1);
v=zeros(n,2);
v(:,1)=x0;
d=zeros(n,2);
g = gradient(f);
c(:,k) = subs(g,x,v(:,k)); % evaluate gradient at initial point


func_eval = 1;
while k < 800
    fprintf('\n*****  step %i:  *****\n',k)
    if mod(k,n)==0 || k==1
        d(:,k) = -c(:,k);
    else
        beta(k) = (norm(c(:,k))/norm(c(:,k-1)))^2; %beta is a number
        d(:,k) = -c(:,k)+beta(k)*d(:,k-1);
    end
    if d(:,k)>1
        d(:,k) = d(:,k)/norm(d(:,k)); %normalization of direction
    end
    
    if order>2
        [alfa_final,FE_GS] = golden_search(f,x,v(:,k),d(:,k),tol_GS,N);
        alfa=vpa(alfa_final,10);
        GS_iteration(k,:)=FE_GS;
    else
        alfa=vpa(solve(diff(subs(f,x,v(:,k)+alf*d(:,k)),alf)==0,alf),12)
        GS_iteration(k,:)=0;
    end

    v(:,k+1) = v(:,k) + alfa*d(:,k);
    c(:,k+1) = subs(g,x,v(:,k+1));
    if(norm(c(:,k+1))<tol_FR)
        
    end
    func_eval=func_eval+1; %function evaluaion 
    fprintf('\nnorm of gradient is: %f  \n\n\n',norm(c(:,k+1)))
    
    if (norm(v(:,k+1)-v(:,k))<tol_FR)
        fprintf('algorithm converged the distance between two opt_point : %f',norm(v(:,k+1)-v(:,k)));
        break;
    end
    k=k+1;
end
fprintf('\n ****************** final step is:%i ****************** \n ',k-1)
fprintf('\nsum of the Fletcher Reeves function evaluation is: %i \n',func_eval)
fprintf('\nsum of the Golden Section evaluation is: %i \n',sum(GS_iteration))
optimum_point = v(:,k)   
optimum_value = vpa(subs(f,x,optimum_point),6)

%% saving Results
GS_iteration(k,:)=0;
v(:,k+1)=[];% due to adding end condition on diffrence between two optimal point we don't have gs iteration for new value so we should ommit it.
x = vpa(v,5)';
FR_iteration=(0:k-1)';

    ResultsTable = table(FR_iteration, GS_iteration);
    for i = 1:n
        ResultsTable.(sprintf('x%d', i)) = double(x(:, i));
    end

    % Save the table to an Excel file without headers
    writetable(ResultsTable, 'Conjugate_Gradient_Results.xlsx')

end