function [optimum_alfa,FE_GS] = golden_search(f,x,x0,d,tol_GS,N)
syms alf
a=0;
b=1;
f(alf)=subs(f, x, x0+ alf*d); % replacing "x0+alfa*d" instead of valriables
g(alf)=f; 
alfa = (-1+sqrt(5))/2;
landa = alfa*a + (1-alfa)*b; % definition of landa
mu = (1-alfa)*a + alfa*b; % definition of mu
g_landa = feval(g,landa);
g_mu = feval(g,mu);
fprintf('------------------------------------------------------\n');
fprintf('    landa        mu        g(landa)      g(mu)      b - a \n');
fprintf('------------------------------------------------------\n');
%fprintf('%.4e %.4e %.4e %.4e %.4e\n', landa, mu, g_landa, g_mu, b-a);
for i = 1:N-2
   if g_landa < g_mu
      b = mu;
      mu = landa;
      g_mu = g_landa;
      landa = a + (1-alfa)*(b-a);
      g_landa = feval(f,landa);
      optimum_alfa=landa;
   else
      a = landa;
      landa = mu;
      g_landa = g_mu;
      mu = a + alfa*(b-a);
      g_mu = feval(f,mu);
      optimum_alfa=mu;
   end;
   
   if (abs(b-a) < tol_GS)
      fprintf('%.4e , %.4e , %.4e , %.4e , %.4e\n', landa, mu, g_landa, g_mu, b-a);
      fprintf('\nGolden sraech succeeded after %d steps and optimum alfa is: %.6e\n', i,optimum_alfa);
      FE_GS=i;
      return;
   end;
end;

fprintf('failed requirements after %d steps\n', N);

end
