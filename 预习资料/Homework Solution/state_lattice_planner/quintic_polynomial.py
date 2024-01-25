import math
import numpy as np

# TODO: implement QuinticPolynomial Solver here
class QuinticPolynomial:
    def __init__(self, x_s, v_s, a_s, x_e, v_e, a_e, T):
        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])

        b = np.array([x_e - x_s - v_s * T - a_s * T ** 2 / 2,
                      v_e - v_s - a_s * T,
                      a_e - a_s])

        X = np.linalg.solve(A, b)

        self.a_s = x_s
        self.a_e = v_s
        self.a2 = a_s / 2.0
        self.a3 = X[0]
        self.a4 = X[1]
        self.a5 = X[2]

    def eval_x(self, t):
        # TODO: evaluation x value given t
        xt = self.a_s + self.a_e * t + self.a2 * t ** 2 + \
                self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def eval_dx(self, t):
        # TODO: evaluation x' value given t
        dxt = self.a_e + 2 * self.a2 * t + \
            3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return dxt

    def eval_ddx(self, t):
        # TODO: evaluation x'' value given t
        ddxt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return ddxt

    def eval_dddx(self, t):
        # TODO: evaluation x''' value given t
        dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return dddxt

