import math
import numpy as np
import osqp
from scipy import sparse

EPLISON = 1e-3

class PathOptimizer:
    def __init__(self, point_nums):
        # the number of path points
        self.point_nums = point_nums

        # the number of optimization variables
        self.variable_nums = 3 * point_nums # l, l', l''
        
        # cost weights
        self.w_l = 0.0
        self.w_dl = 0.0
        self.w_ddl = 0.0
        self.w_dddl = 0.0
        self.w_ref_l = 0.0

        # the interval between two points
        self.step_list = []
        self.step_sqr_list = []

        # the reference l list for optimizer
        self.ref_l_list = []

        # initial and end states
        self.init_state = []
        self.end_state = []

        # state bounds 
        self.l_upper_bound = []
        self.l_lower_bound = []
        self.dl_upper_bound = []
        self.dl_lower_bound = []
        self.ddl_upper_bound = []
        self.ddl_lower_bound = []
        self.dddl_upper_bound = []
        self.dddl_lower_bound = []

        # the solution of optimizer
        self.solution_l = []
        self.solution_dl = []
        self.solution_ddl = []
        self.solution_dddl = []
        self.solution_theta = []
        self.solution_kappa = []
        self.solution_dkappa = []
        
    def SetCostingWeights(self, w_l, w_dl, w_ddl, w_dddl, w_ref_l):
        self.w_l = w_l
        self.w_dl = w_dl
        self.w_ddl = w_ddl
        self.w_dddl = w_dddl
        self.w_ref_l = w_ref_l

    def SetReferenceLList(self, ref_l_list):
        self.ref_l_list = ref_l_list

    def SetStepList(self, step_list):
        self.step_list = step_list
        self.step_sqr_list = [x for x in self.step_list]

    def SetLBound(self, upper_bound, lower_bound):
        self.l_lower_bound = lower_bound
        self.l_upper_bound = upper_bound

    def SetDlBound(self, upper_bound, lower_bound):
        self.dl_lower_bound = lower_bound
        self.dl_upper_bound = upper_bound

    def SetDdlBound(self, upper_bound, lower_bound):
        self.ddl_lower_bound = lower_bound
        self.ddl_upper_bound = upper_bound

    def SetDddlBound(self, upper_bound, lower_bound):
        self.dddl_lower_bound = lower_bound
        self.dddl_upper_bound = upper_bound

    def SetInitState(self, init_state):
        self.init_state = init_state

    def SetEndState(self, end_state):
        self.end_state = end_state

    def FormulateMatrixP(self):
        # TODO: Construct matrix P for objective function.
        return P

    def FormulateVectorq(self):
        # TODO: Construct vector q for objective function.
        return q

    def FormulateAffineConstraint(self):
        # TODO: Construct matrix A and vector l, u for constraints.
        return A, lb, ub

    def Solve(self):
        # TODO: 1. Construct QP problem (P, q, A, l, u)
        # refer to sparse.csc_matrix doc:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
        P = self.FormulateMatrixP()
        q = self.FormulateVectorq()

        A, lb, ub = self.FormulateAffineConstraint()


        # TODO: 2. Create an OSQP object and solve 
        # please refer to https://osqp.org/docs/examples/setup-and-solve.html
        

        # TODO: 3. Extract solution from osqp result
        

    def GetSolution(self):
        return self.solution_l, self.solution_dl, self.solution_ddl, self.solution_dddl

# sparse.csc_matrix doc:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
# example:
# row = [0, 2, 2, 0, 1, 2, 0, 0, 2]
# col = [0, 0, 1, 2, 2, 2, 0, 1, 0]
# data = [1, 2, 3, 4, 5, 6, 3, 8, 2]
# print(sparse.csc_matrix((data, (row, col)), shape=(3, 3)).toarray())