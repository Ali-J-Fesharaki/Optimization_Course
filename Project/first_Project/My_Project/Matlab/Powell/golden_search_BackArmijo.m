function [optimum_alfa,FE_GS] = golden_search_BackArmijo(F_al,tol_GS,N)
format long
syms alf
g(alf)=F_al; % replacing "x0+alfa*d" instead of valriables
a=0;
b=1;
%% Armijo
dg0=vpa(subs(diff(g,alf),alf,0),5); % one function evaluation
ro=0.2;
eta=2;
FE_Arm=2;
if dg0<0 % condition for using Armijo : diff(g(alfa)) <0 @alfa=0
    q(alf)=vpa(subs(g,alf,0)+alf*(ro*subs(diff(g,alf),alf,0)));
    alf_Arm=1;
    j=1;
    g_Arm(j,:)=vpa(subs(g,alf,alf_Arm),5);
    q_Arm(j,:)=vpa(subs(q,alf,alf_Arm),5);
    FE_Arm=4;
    if g_Arm(j,:)> q_Arm(j,:)
     while g_Arm(j,:)> q_Arm(j,:)
        j=j+1;
        alf_Arm(j,:)=alf_Arm(j-1,:)/eta;
        g_Arm(j,:)=vpa(subs(g,alf,alf_Arm(j,:)),3);
        q_Arm(j,:)=vpa(subs(q,alf,alf_Arm(j,:)),3);
        FE_Arm=FE_Arm+2;
        a=alf_Arm(j,:);
        b=alf_Arm(j-1,:);
     end
    else %g_Arm(j,:) < q_Arm(j,:)
        optimum_alfa=0.9999
        FE_GS=FE_Arm;
        return
    end
else 
    optimum_alfa=0.0001
    FE_GS=FE_Arm;
    return
end


%% Golden Section
alfa = (-1+sqrt(5))/2;
landa = alfa*a + (1-alfa)*b; % definition of landa
mu = (1-alfa)*a + alfa*b; % definition of mu
g_landa = vpa(subs(g,alf,landa));
g_mu = vpa(subs(g,alf,mu));
% 2 iteration
fprintf('------------------------------------------------------\n');
fprintf('    landa        mu        g(landa)      g(mu)      b - a \n');
fprintf('------------------------------------------------------\n');
%fprintf('%.4e %.4e %.4e %.4e %.4e\n', landa, mu, g_landa, g_mu, b-a);
for i = 1:N-2
   if g_landa < g_mu;
      b = mu;
      mu = landa;
      g_mu = g_landa;
      landa = a + (1-alfa)*(b-a);
      g_landa = vpa(subs(g,alf,landa));
      optimum_alfa=landa;
   else
      a = landa;
      landa = mu;
      g_landa = g_mu;
      mu = a + alfa*(b-a);
      g_mu = vpa(subs(g,alf,mu));
      optimum_alfa=mu;
   end;
%  fprintf('%.4e %.4e %.4e %.4e %.4e\n', landa, mu, g_landa, g_mu, b-a);
   if (abs(b-a) < tol_GS)
       fprintf('%.4e , %.4e ,  %.4e ,  %.4e ,  %.4e\n', landa, mu, g_landa, g_mu, b-a);
       FE_GS=i+3+FE_Arm;
       fprintf('\nLine Search succeeded after %d steps and optimum alfa is: %.6e\n', FE_GS,optimum_alfa);
       return;
   end;
end;
fprintf('failed requirements after %d steps\n', N);

end
