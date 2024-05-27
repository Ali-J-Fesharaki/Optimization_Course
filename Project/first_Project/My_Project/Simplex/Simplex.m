function [Optimum_Point,Optimum_value] = Simplex(f,x,x0,tol_simplex)
% Assumption:
global FE FE_list;
alfa_Q = -0.5;
delta = 0.5;
alfa_E = 3;
alfa_R = 1;
Operation = strings();
n=size(x0,1);
S(:,:,1) = zeros(n,n+1);
for j=1:n+1
    S(:,j,1) = x0;
end
v(:,:,1) = S(:,:,1) + delta*[zeros(n,1) , eye(n,n)]; %first simplex
for j=1:n+1
    F(1,j) =FEC(f,x,v(:,j,1)); 
end
for s=1:300
    for j=1:n+1
        [m, ind]=min(F(s,:));
        F_sorted(s,j) = m; % sorting values of cost function
        x_sorted(:,j,s) = v(:,ind,s); % sorting variables
        F(s,ind) = NaN;
        v(:,ind,s) = NaN;
    end
    
    x_w(:,s) = x_sorted(:,n+1,s); %the worst x
    
    for j=1:n+1
        v(:,j,s) = x_sorted(:,j,s); %sorting variables in cardinal matrix
        F(s,j) = F_sorted(s,j); %sorting values of f in cardinal vector
    end
    
    x_sum(:,s) = zeros(n,1);
    for j=1:n
        x_sum(:,s) = x_sum(:,s) + v(:,j,s); %sumation of appropriate variables in simplex s
    end
    x_c(:,s) = x_sum(:,s)/n; %mean of simplex without the worst point
    for j=1:n+1
        dis(s,j) = norm(v(:,j,s)-x_c(:,s)); % distance between all variable and mean point
    end
    if mean(dis(s,:)) < tol_simplex % exit criteria
        break
    end
    x_R(:,s) = (1+alfa_R)*x_c(:,s) - alfa_R*x_w(:,s) %Reflection
    F_R(s,:) =FEC(f,x,x_R(:,s));
    
    if F_R(s,:)<F(s,n)&&F_R(s,:)>=F(s,1)
        Operation(s,:) = 'Reflection';
        v(:,n+1,s+1) = x_R(:,s);
        F(s+1,n+1) = F_R(s,:);
        for j=1:n
            v(:,j,s+1) = v(:,j,s);
            F(s+1,j) = F(s,j);
        end      
    elseif F_R(s,:)>=F(s,n)     
        if F_R(s,:)<F(s,n+1)&&F_R(s,:)>=F(s,n)
            
            x_Q(:,s) = (1+alfa_Q)*x_R(:,s) - alfa_Q*x_c(:,s) %outside contraction
            F_Q(s,:) = FEC(f,x,x_Q(:,s));
            if F_Q(s,:)<F(s,n)
                Operation(s,:) = 'Outside Contraction';
                v(:,n+1,s+1) = x_Q(:,s);
                F(s+1,n+1) = F_Q(s,:);
                for j=1:n
                    v(:,j,s+1) = v(:,j,s);
                    F(s+1,j) = F(s,j);
                end
            else
                Operation(s,:) = 'Shrinking';
                for j=2:n+1
                    v(:,j,s+1) = v(:,1,s) + delta*(v(:,j,s)-v(:,1,s)) %shrinking
                    F(s+1,j) = FEC(f,x,v(:,j,s+1));
                end
                v(:,1,s+1) = v(:,1,s);
                F(s+1,1) = F(s,1);
            end
            
        else
            
            x_Q(:,s) = (1+alfa_Q)*x_c(:,s) - alfa_Q*x_w(:,s) %inside contraction
            F_Q(s,:) = FEC(f,x,x_Q(:,s));
            if F_Q(s,:)<F(s,n+1)
                Operation(s,:) = 'Inside Contraction';
                v(:,n+1,s+1) = x_Q(:,s);
                F(s+1,n+1) = F_Q(s,:);
                for j=1:n
                    v(:,j,s+1) = v(:,j,s);
                    F(s+1,j) = F(s,j);
                end
            else
                Operation(s,:) = 'Shrinking';
                for j=2:n+1
                    v(:,j,s+1) = v(:,1,s) + delta*(v(:,j,s)-v(:,1,s)) %shrinking
                    F(s+1,j) = FEC(f,x,v(:,j,s+1));
                end
                v(:,1,s+1) = v(:,1,s);
                F(s+1,1) = F(s,1);
            end
            
        end  
        
    else
        x_E(:,s) = (1+alfa_E)*x_c(:,s) - alfa_E*x_w(:,s) %expansion
        F_E(s,:) = FEC(f,x,x_E(:,s));
        if F_E(s,:)<F_R(s,:)
            Operation(s,:) = 'Expansion';
            v(:,n+1,s+1) = x_E(:,s);
            F(s+1,n+1) = F_E(s,:);
            for j=1:n
                v(:,j,s+1) = v(:,j,s);
                F(s+1,j) = F(s,j);
            end
        else
            Operation(s,:) = 'Reflection';
            v(:,n+1,s+1) = x_R(:,s);
            F(s+1,n+1) = F_R(s,:);
            for j=1:n
                v(:,j,s+1) = v(:,j,s);
                F(s+1,j) = F(s,j);
            end
        end

    end
        
end

%% saving Results
iteration = [0:1:s]';
x1(1:s,:) = v(1,1,:);
x1(s+1,:) = v(1,1,s);
x2(1:s,:) = v(2,1,:);
x2(s+1,:) = v(2,1,s);
if n==3
    x3(1:s,:) = v(3,1,:);
    x3(s+1,:) = v(3,1,s);
end
FunctionValue(1:s,:) = double(F(:,1));
FunctionValue(s+1,:) = double(F(s,1));
Procedure = strings();
Procedure(1,:) = 'initial simplex';
Procedure(2:s,:) = Operation;
Procedure(s+1,:) = '-';
if n==2
    Results = table(iteration, x1, x2, FunctionValue, Procedure);
elseif n==3
    Results = table(iteration, x1, x2, x3, FunctionValue, Procedure);
end
filename = 'Simplex_Results.xlsx';
writetable(Results,filename)

iteration = s-1
Function_Evaluaion = FE
Optimum_Point = v(:,1,s);
Optimum_value=vpa(subs(f,x,v(:,1,s)));


