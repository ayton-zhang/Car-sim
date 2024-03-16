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
        rows = []
        cols = []
        data = []

        # l cost
        for i in range(self.point_nums):
            rows.append(i)
            cols.append(i)
            data.append(self.w_l)

        # dl cost
        for i in range(self.point_nums):
            rows.append(i + self.point_nums)
            cols.append(i + self.point_nums)
            data.append(self.w_dl)

        # ddl cost
        for i in range(self.point_nums):
            rows.append(i + 2 * self.point_nums)
            cols.append(i + 2 * self.point_nums)
            data.append(self.w_ddl)

        # dddl cost
        for i in range(self.point_nums - 1):
            rows.append(i + 2 * self.point_nums)
            cols.append(i + 2 * self.point_nums)
            data.append(self.w_dddl / self.step_sqr_list[i])
            
            rows.append(i + 1 + 2 * self.point_nums)
            cols.append(i + 1 + 2 * self.point_nums)
            data.append(self.w_dddl / self.step_sqr_list[i])

            rows.append(i + 1 + 2 * self.point_nums)
            cols.append(i + 2 * self.point_nums)
            data.append(-2.0 * self.w_dddl / self.step_sqr_list[i])

        # ref l cost
        for i in range(self.point_nums):
            rows.append(i)
            cols.append(i)
            data.append(self.w_ref_l)

        for i in range(len(data)):
            data[i] *= 2.0
        
        P = sparse.csc_matrix((data, (rows, cols)), shape=(self.variable_nums, self.variable_nums))

        return P

    def FormulateVectorq(self):
        # TODO: Construct vector q for objective function.
        q = np.zeros(self.variable_nums)
        for i in range(self.point_nums):
            q[i] = -2.0 * self.w_ref_l * self.ref_l_list[i]
        return q

    def FormulateAffineConstraint(self):
        # TODO: Construct matrix A and vector l, u for constraints.
        # Initial and end state constraints
        l = []
        u = []
        constraint_idx = 0
        cols = []
        rows = []
        data = []

        # l boudary constraints
        for i in range(self.point_nums):
            rows.append(constraint_idx)
            cols.append(i)
            data.append(1.0)
            l.append(self.l_lower_bound[i])
            u.append(self.l_upper_bound[i])
            constraint_idx += 1
        # dl boudary constraints
        for i in range(self.point_nums):
            rows.append(constraint_idx)
            cols.append(self.point_nums + i)
            data.append(1.0)
            l.append(self.dl_lower_bound[i])
            u.append(self.dl_upper_bound[i])
            constraint_idx += 1
        # ddl boudary constraints
        for i in range(self.point_nums):
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i)
            data.append(1.0)
            l.append(self.ddl_lower_bound[i])
            u.append(self.ddl_upper_bound[i])
            constraint_idx += 1
        # dddl boudary constraints
        for i in range(self.point_nums - 1):
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i)
            data.append(-1)
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i + 1)
            data.append(1)
            l.append(-self.dddl_lower_bound[i] * self.step_list[i])
            u.append(self.dddl_upper_bound[i] * self.step_list[i])
            constraint_idx += 1


        # initial l state constraints
        rows.append(constraint_idx)
        cols.append(0)
        data.append(1.0)
        l.append(self.init_state[0] - EPLISON)
        u.append(self.init_state[0] + EPLISON)
        constraint_idx += 1
        # initial dl state constraints
        rows.append(constraint_idx)
        cols.append(self.point_nums)
        data.append(1.0)
        l.append(self.init_state[1] - EPLISON)
        u.append(self.init_state[1] + EPLISON)
        constraint_idx += 1
        # initial ddl state constraints
        rows.append(constraint_idx)
        cols.append(2 * self.point_nums)
        data.append(1.0)
        l.append(self.init_state[2] - EPLISON)
        u.append(self.init_state[2] + EPLISON)
        constraint_idx += 1


        # end l state constraints
        rows.append(constraint_idx)
        cols.append(self.point_nums - 1)
        data.append(1.0)
        l.append(self.end_state[0] - EPLISON)
        u.append(self.end_state[0] + EPLISON)
        constraint_idx += 1
        # end dl state constraints
        rows.append(constraint_idx)
        cols.append(2 * self.point_nums - 1)
        data.append(1.0)
        l.append(self.end_state[1] - EPLISON)
        u.append(self.end_state[1] + EPLISON)
        constraint_idx += 1
        # end ddl state constraints
        rows.append(constraint_idx)
        cols.append(3 * self.point_nums - 1)
        data.append(1.0)
        l.append(self.end_state[2] - EPLISON)
        u.append(self.end_state[2] + EPLISON)
        constraint_idx += 1


        # dl continuity constraints
        for i in range(self.point_nums -1):
            rows.append(constraint_idx)
            cols.append(self.point_nums + i)
            data.append(-1)
            rows.append(constraint_idx)
            cols.append(self.point_nums + i + 1)
            data.append(1)
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i)
            data.append(-0.5 * self.step_list[i])
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i + 1)
            data.append(-0.5 * self.step_list[i])
            l.append(1e-5)
            u.append(1e-5)
            constraint_idx += 1
        # l continuity constraints
        for i in range(self.point_nums - 1):
            rows.append(constraint_idx)
            cols.append(i)
            data.append(-1)
            rows.append(constraint_idx)
            cols.append(i + 1)
            data.append(1)
            rows.append(constraint_idx)
            cols.append(self.point_nums + i)
            data.append(-self.step_list[i])
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i)
            data.append(-1/3 * self.step_sqr_list[i])
            rows.append(constraint_idx)
            cols.append(2 * self.point_nums + i + 1)
            data.append(-1/6 * self.step_sqr_list[i])
            l.append(1e-5)
            u.append(1e-5)
            constraint_idx += 1
        
        A = sparse.csc_matrix((data, (rows, cols)), shape=(constraint_idx, self.variable_nums))

        return A, l, u

    def Solve(self):
        # TODO: 1. Construct QP problem (P, q, A, l, u)
        # refer to sparse.csc_matrix doc:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
        P = self.FormulateMatrixP()
        q = self.FormulateVectorq()

        A, l, u = self.FormulateAffineConstraint()


        # TODO: 2. Create an OSQP object and solve 
        # please refer to https://osqp.org/docs/examples/setup-and-solve.html
        osqp_problem = osqp.OSQP()
        osqp_problem.setup(P, q, A, l, u, polish=True, eps_abs=1e-5, eps_rel=1e-5,
                            eps_prim_inf=1e-5, eps_dual_inf=1e-5, verbose=True)

        # setting warmstart for l, l', l''
        var_warm_start = np.array(
            self.ref_l_list + [0.0 for n in range(2 * self.point_nums)])
        osqp_problem.warm_start(x=var_warm_start)

        # solve
        res = osqp_problem.solve()

        # TODO: 3. Extract solution from osqp result
        self.solution_l = res.x[0:self.point_nums]
        self.solution_dl = res.x[self.point_nums : 2 * self.point_nums]
        self.solution_ddl = res.x[2 * self.point_nums : 3 * self.point_nums]

        for i in range(self.point_nums - 1):
            self.solution_dddl.append(
                (self.solution_ddl[i + 1] - self.solution_ddl[i]) / self.step_list[i])
        self.solution_dddl.append(0.0)

    def GetSolution(self):
        return self.solution_l, self.solution_dl, self.solution_ddl, self.solution_dddl

# sparse.csc_matrix doc:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
# example:
# row = [0, 2, 2, 0, 1, 2, 0, 0, 2]
# col = [0, 0, 1, 2, 2, 2, 0, 1, 0]
# data = [1, 2, 3, 4, 5, 6, 3, 8, 2]
# print(sparse.csc_matrix((data, (row, col)), shape=(3, 3)).toarray())