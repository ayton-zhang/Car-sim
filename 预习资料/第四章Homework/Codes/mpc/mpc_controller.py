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

        # Easy NLP modeling in CasADi with Opti：https://web.casadi.org/blog/opti/
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


        # TODO: apply bicycle model dynamic constraints


        # TODO: apply state/control boundary constraints
        pass


    def cal_quadratic_form(self, x, Q):
        return casadi.mtimes(x, casadi.mtimes(Q, x.T))

    def calculate_cost(self):
        # TODO: calculate cost function
        pass

    def solve(self):
        # TODO: solve this NMPC problem and return the solution
        pass

    def update_initial_condition(self, x, y, yaw, vel):
        self.opti.set_value(self.initial_state, [x, y, yaw, vel])

    def update_reference(self, x_ref, y_ref, yaw_ref, v_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.yaw_ref, yaw_ref)
        self.opti.set_value(self.v_ref,   v_ref)

    def update_previous_input(self, last_acc, last_delta):
        self.opti.set_value(self.last_cycle_control, [last_acc, last_delta])
