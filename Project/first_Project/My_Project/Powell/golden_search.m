function [optimum_alfa,FE_GS] = golden_search(F_al,tol_GS,N)
format long
syms alf
g(alf)=F_al; % replacing "x0+alfa*d" instead of valriables
a=0;
b=1;

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
       FE_GS=i+3;
       fprintf('\nLine Search succeeded after %d steps and optimum alfa is: %.6e\n', FE_GS,optimum_alfa);
       return;
   end;
end;
fprintf('failed requirements after %d steps\n', N);

end





