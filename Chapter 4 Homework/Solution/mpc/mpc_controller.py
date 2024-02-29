import casadi
# CasADi website: https://web.casadi.org/

WHEEL_BASE = 2.5  # [m]

MIN_VEL = 0.0    # min/max velocity
MAX_VEL = 40.0
MIN_ACC = -3.0   # min/max acceleration
MAX_ACC = 3.0
MIN_DELTA = -1.0   # min/max front steer angle
MAX_DELTA = 1.0
MIN_JERK = -1.5   # min/max jerk
MAX_JERK = 2.0
MIN_OMEGA = -0.5   # min/max front steer angle rate
MAX_OMEGA = 0.5

Q = [5., 5., 10., 50.0]  # weights on x, y, yaw, and v
R = [1., 1000.]  # weights on jerk and omega

class MPCTrajectoryTracker():

    def __init__(self,
                 N=7,     # timesteps in MPC Horizon
                 DT=0.2):

        self.N = N + 1
        self.DT = DT

        self.opti = casadi.Opti()

        # 1. setting "patameter" for CasADi
        # In CasADi, "parameter" can not be changed during optimization process
        # they are fixed values provided externally.
        # last cycle optimal control
        self.last_cycle_control = self.opti.parameter(2)
        # current cycle initial state
        self.initial_state = self.opti.parameter(4)

        # reference trajectory to tracking
        self.ref_states = self.opti.parameter(self.N, 4)
        self.x_ref = self.ref_states[:, 0]
        self.y_ref = self.ref_states[:, 1]
        self.yaw_ref = self.ref_states[:, 2]
        self.v_ref = self.ref_states[:, 3]

        # 2. setting "variable" for CasADi
        # In CasADi, "variable" is the variable that needs to be optimized
        # states <[x, y, yaw, v]_0, ... [x, y, yaw, v]_N>
        self.opt_state_x = self.opti.variable(self.N+1, 4)
        self.opt_x = self.opt_state_x[:, 0]
        self.opt_y = self.opt_state_x[:, 1]
        self.opt_yaw = self.opt_state_x[:, 2]
        self.opt_v = self.opt_state_x[:, 3]

        # control inputs <[acc, delta]_0, ... [acc, delta]_N-1>
        self.opt_input_u = self.opti.variable(self.N, 2)
        self.opt_acc = self.opt_input_u[:, 0]
        self.opt_delta = self.opt_input_u[:, 1]

        # 3. setting solver
        self.opti.solver('ipopt')

        # 4. contruct cost function and eq/ieq constraints
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.calculate_cost()
        self.construct_constraints()

    def construct_constraints(self):
        # TODO: apply initial state constraints
        self.opti.subject_to(self.opt_x[0] == self.initial_state[0])
        self.opti.subject_to(self.opt_y[0] == self.initial_state[1])
        self.opti.subject_to(self.opt_yaw[0] == self.initial_state[2])
        self.opti.subject_to(self.opt_v[0] == self.initial_state[3])

        # TODO: apply bicycle model dynamic constraints
        for i in range(self.N):
            self.opti.subject_to(
                self.opt_x[i+1] == self.opt_x[i] + self.DT * (self.opt_v[i] * casadi.cos(self.opt_yaw[i])))
            self.opti.subject_to(
                self.opt_y[i+1] == self.opt_y[i] + self.DT * (self.opt_v[i] * casadi.sin(self.opt_yaw[i])))
            self.opti.subject_to(self.opt_yaw[i+1] == self.opt_yaw[i] + self.DT * (
                self.opt_v[i] / WHEEL_BASE * casadi.tan(self.opt_delta[i])))
            self.opti.subject_to(
                self.opt_v[i+1] == self.opt_v[i] + self.DT * (self.opt_acc[i]))

        # TODO: apply state/control boundary constraints
        self.opti.subject_to(self.opti.bounded(MIN_VEL, self.opt_v, MAX_VEL))

        self.opti.subject_to(self.opti.bounded(
            MIN_ACC,  self.opt_acc, MAX_ACC))

        self.opti.subject_to(self.opti.bounded(
            MIN_DELTA, self.opt_delta,  MAX_DELTA))

        self.opti.subject_to(self.opti.bounded(MIN_JERK*self.DT, self.opt_acc[0] - self.last_cycle_control[0],
                                               MAX_JERK*self.DT))

        self.opti.subject_to(self.opti.bounded(MIN_OMEGA*self.DT, self.opt_delta[0] - self.last_cycle_control[1],
                                               MAX_OMEGA*self.DT))

        for i in range(self.N - 1):
            self.opti.subject_to(self.opti.bounded(MIN_JERK*self.DT, self.opt_acc[i+1] - self.opt_acc[i],
                                                   MAX_JERK*self.DT))

            self.opti.subject_to(self.opti.bounded(MIN_OMEGA*self.DT, self.opt_delta[i+1] - self.opt_delta[i],
                                                   MAX_OMEGA*self.DT))

    def cal_quadratic_form(self, x, Q):
        return casadi.mtimes(x, casadi.mtimes(Q, x.T))

    def calculate_cost(self):
        total_cost = 0
        # TODO: calculate cost function
        for i in range(self.N):
            # reference states tracking cost
            total_cost += self.cal_quadratic_form(self.opt_state_x[i+1, :] -
                                                  self.ref_states[i, :], self.Q)

        for i in range(self.N - 1):
            # input smoothness cost
            total_cost += self.cal_quadratic_form(self.opt_input_u[i+1, :] -
                                                  self.opt_input_u[i, :], self.R)
        self.opti.minimize(total_cost)

    def solve(self):
        # TODO: solve this NMPC problem and return the solution
        try:
            solution = self.opti.solve()
            # Optimal solution.
            mpc_us = solution.value(self.opt_input_u)
            mpc_xs = solution.value(self.opt_state_x)
            ref_states = solution.value(self.ref_states)
        except:
            # Suboptimal solution (e.g. timed out).
            mpc_us = self.opti.debug.value(self.opt_input_u)
            mpc_xs = self.opti.debug.value(self.opt_state_x)
            ref_states = self.opti.debug.value(self.ref_states)

        result = {}
        # only the first step of the control strategy is implemented
        result['implemented_u'] = mpc_us[0, :]
        # solution inputs
        result['mpc_us'] = mpc_us
        # solution states
        result['mpc_xs'] = mpc_xs
        # input reference states
        result['ref_states'] = ref_states

        return result

    def update_initial_condition(self, x, y, yaw, vel):
        self.opti.set_value(self.initial_state, [x, y, yaw, vel])

    def update_reference(self, x_ref, y_ref, yaw_ref, v_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.yaw_ref, yaw_ref)
        self.opti.set_value(self.v_ref,   v_ref)

    def update_previous_input(self, last_acc, last_delta):
        self.opti.set_value(self.last_cycle_control, [last_acc, last_delta])
