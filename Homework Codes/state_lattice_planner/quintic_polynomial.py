import math
import numpy as np

# TODO: implement QuinticPolynomial Solver here
class QuinticPolynomial:
    def __init__(self, x_s, v_s, a_s, x_e, v_e, a_e, T):
        self.x_s = x_s
        self.v_s = v_s
        self.a_s = a_s
        self.x_e = x_e
        self.v_e = v_e
        self.a_e = a_e
        self.T = T

    def eval_x(self, t):
        # TODO: evaluation x value given t
        xt = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        return xt

    def eval_dx(self, t):
        # TODO: evaluation x' value given t
        return dxt

    def eval_ddx(self, t):
        # TODO: evaluation x'' value given t
        return ddxt

    def eval_dddx(self, t):
        # TODO: evaluation x''' value given t
        return dddxt

