#! /usr/bin/python3

##--------------------------------------------------------------------\
#   direct_python
#   './direct_python/src/main_test_details.py'
#   Test function/example for using the DIRECT optimizer. Format updates
#       are for integration in the AntennaCAT GUI.
#
#   Author(s): Lauren Linkous
#   Last update: June 11, 2026
##--------------------------------------------------------------------\

import numpy as np
import pandas as pd
import time

from direct import direct

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs     # multi objective function


class TestDetails():
    def __init__(self):
        # Constant variables
        E_TOL = 10 ** -4      # Convergence Error Tolerance
        MAXIT = 3000          # Maximum allowed objective function calls

        # Objective function dependent variables
        LB = func_configs.LB              # Lower boundaries, [[0.21, 0, 0.1]]
        UB = func_configs.UB              # Upper boundaries, [[1, 1, 0.5]]
        TARGETS = func_configs.TARGETS    # Target values for output

        # threshold is same dims as TARGETS
        # 0 = use target value as actual target. value should EQUAL target
        # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
        # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
        # DEFAULT THRESHOLD
        THRESHOLD = np.zeros_like(TARGETS)
        evaluate_threshold = False

        # Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC   # objective function
        constr_F = func_configs.CONSTR_FUNC    # constraint function

        # optimizer specific vars
        EPS = 1e-4            # Jones' epsilon. balance of local vs global search

        # optimizer setting values
        self.best_eval = 1            # Starting eval value

        parent = self                 # Optional parent class for optimizer
                                        # (Used for passing debug messages or
                                        # other information that will appear
                                        # in GUI panels)

        self.suppress_output = True   # Suppress the console output of the optimizer

        self.allow_update = True      # Allow objective call to update state

        # instantiation of DIRECT optimizer
        opt_params = {'EPS': [EPS]}
        opt_df = pd.DataFrame(opt_params)
        self.myDirect = direct(LB, UB, TARGETS, E_TOL, MAXIT,
                               func_F, constr_F,
                               opt_df,
                               parent=parent,
                               evaluate_threshold=evaluate_threshold,
                               obj_threshold=THRESHOLD)


    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)


    def run(self):

        last_iter = 0
        while not self.myDirect.complete():

            # step through optimizer processing
            # consumes the previous evaluation and advances the partition
            self.myDirect.step(self.suppress_output)

            # call the objective function, control
            # when it is allowed to update and return
            # control to the optimizer
            noErr = self.myDirect.call_objective(self.allow_update)
            if noErr == True:
                iter, eval = self.myDirect.get_convergence_data()
                if (eval < self.best_eval) and (eval != 0):
                    self.best_eval = eval
                if iter > last_iter:
                    last_iter = iter
                    if self.suppress_output:
                        if iter % 200 == 0:
                            print("************************************************")
                            print("Objective Function Iterations: " + str(iter))
                            print("Best Eval: " + str(self.best_eval))
            else:
                print("ERROR: in executing objective function call.")

        print("************************************************")
        print("Total Objective Function Iterations: " + str(last_iter))
        print("Best Eval: " + str(self.best_eval))
        print("Optimized Solution")
        print(self.myDirect.get_optimized_soln())
        print("Optimized Outputs")
        print(self.myDirect.get_optimized_outs())



if __name__ == "__main__":
    dr = TestDetails()
    dr.run()