from direct_methods import NelderMead
from gradient_methods import FletcherReeves
from functions import f_1 ,f_2,grad_f1,grad_f2

if(__name__=="__main__"):
    optimum_point, optimum_value, func_eval,ls,iter_count, df=FletcherReeves(f_1,grad_f=grad_f1,tol_ls=1e-4,tol=1e-4).optimize([0,0,0])
    print("Optimum point:",optimum_point)
    print("Optimum value:",optimum_value)
    print("Function evaluation count:",func_eval)
    print("Line search evaluation count:",ls)
    print("Number of iterations:",iter_count)