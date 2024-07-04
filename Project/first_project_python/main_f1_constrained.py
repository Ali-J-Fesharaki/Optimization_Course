from functions import constrained_f1 , constraint1_f1,constraint2_f1,constraint3_f1
from classic_constraint_methods import Extrior_Penalty,Augmented_Lagrangian,SQP_Optimizer
import tabulate
import numpy as np
if (__name__ == "__main__"):
    X0= np.array([1.8,1.7,1.9,-0.8,-0.8])
    optimizer_1=Extrior_Penalty(constrained_f1,eq_constraints=[constraint1_f1,constraint2_f1,constraint3_f1])
    optimizer_2 = Augmented_Lagrangian(constrained_f1,eq_constraints=[constraint1_f1,constraint2_f1,constraint3_f1])
    optimizer_3 = SQP_Optimizer(constrained_f1,eq_constraints=[constraint1_f1,constraint2_f1,constraint3_f1])
    x_1,func_eval_1,iter_count_1,df_1 = optimizer_1.optimize(X0)
    print("Extrior Penalty Method_finished")
    x_2,func_eval_2,iter_count_2,df_2= optimizer_2.optimize(X0)
    print("Augmented Lagrangian Method_finished")
    x_3,func_eval_3,iter_count_3,df_3 = optimizer_3.optimize(X0)
    print("SQP Method_finished")

    print(f"Extrior Penalty Method: {x_1} : {func_eval_1} : {iter_count_1}")
    print(f"Augmented Lagrangian Method: {x_2} : {func_eval_2} : {iter_count_2}")
    print(f"SQP Method: {x_3} : {func_eval_3} : {iter_count_3}")
