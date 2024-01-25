import math
import numpy as np
import osqp
from scipy import sparse
from matplotlib import pyplot as plt
import sys
import pathlib
sys.path.append(str(pathlib.Path.cwd().parent))
from utils.path_utils import *
from utils.vehicle_utils import WHEEL_BASE, MAX_STEER, MAX_DSTEER
from frenet_path_optimizer import PathOptimizer

# cost weight parameters
WEIGHT_L = 1.0
WEIGHT_DL = 100.0
WEIGHT_DDL = 1000.0
WEIGHT_DDDL = 10000.0
WEIGHT_REF_L = 1.0

DL_LIMIT = 0.6
DDL_LIMIT = 0.4
DDDL_LIMIT = 0.1

def generate_reference_line_and_boundary_1():
    # generate reference line
    key_pt_xs = [10, 70]
    key_pt_ys = [10, 10]
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, ref_path_s_list = \
            generate_cublic_spline_reference_line(key_pt_xs, key_pt_ys, ds=1.0)

    boundary_left_dists = np.array([])
    boundary_right_dists = np.array([])
    for i in range(len(ref_path_s_list)):
        boundary_left_dists = np.append(
            boundary_left_dists, 0.0 + 2.0)
        boundary_right_dists = np.append(
            boundary_right_dists, 0.0 - 2.0)
    # print(len(ref_path_s_list))
    for i in range(len(ref_path_s_list)):
        if i >= 10 and i <= 15:
            boundary_left_dists[i] = 0.5
        if i >= 30 and i <= 35:
            boundary_right_dists[i] = -0.5
        if i >= 45 and i <= 50:
            boundary_left_dists[i] = 0.5

    return ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
                 ref_path_s_list, boundary_left_dists, boundary_right_dists

def generate_reference_line_and_boundary_2():
    # generate reference line
    # key_pt_xs = [10, 10, 20, 60, 65, 30, 40, 80]
    # key_pt_ys = [10, 80, 80, 90, 60, 40, 20, 10]
    key_pt_xs = [10, 10, 20, 60, 65, 50, 40, 37, 40, 80]
    key_pt_ys = [10, 80, 80, 90, 45, 40, 70, 70, 20, 20]
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, ref_path_s_list = \
            generate_cublic_spline_reference_line(key_pt_xs, key_pt_ys, ds=1.0)
    print(len(ref_path_s_list))
    boundary_left_dists = np.array([])
    boundary_right_dists = np.array([])
    for i in range(len(ref_path_s_list)):
        boundary_left_dists = np.append(
            boundary_left_dists, 0.0 + 3.0)
        boundary_right_dists = np.append(
            boundary_right_dists, 0.0 - 3.0)
    for i in range(len(ref_path_s_list)):
        if i >= 10 and i <= 15:
            boundary_left_dists[i] = 0.5
        if i >= 25 and i <= 30:
            boundary_right_dists[i] = -0.5
        if i >= 215 and i <= 220:
            boundary_right_dists[i] = 1.0
        if i >= 300 and i <= 310:
            boundary_left_dists[i] = -0.5

    return ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
                 ref_path_s_list, boundary_left_dists, boundary_right_dists

if __name__ == "__main__":

    # generate reference line and boundary
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
        ref_path_s_list, boundary_left_dists, boundary_right_dists = generate_reference_line_and_boundary_2()

    left_boundary_pt_xs, left_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
            ref_path_x_list,ref_path_y_list, ref_path_yaw_list, boundary_left_dists)
    right_boundary_pt_xs, right_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
            ref_path_x_list,ref_path_y_list, ref_path_yaw_list, boundary_right_dists)
    
    ref_path_size = len(ref_path_s_list)
    ref_path_step_list = [ref_path_s_list[i+1] - ref_path_s_list[i]
                       for i in range(ref_path_size - 1)]
    ref_l_list = [0.5 * (boundary_right_dists[i] +
                         boundary_left_dists[i]) for i in range(ref_path_size)]

    init_state = [ref_l_list[0], 0.0, 0.0]
    end_state = [ref_l_list[-1], 0.0, 0.0]

    # state bounds
    # l bound
    l_lower_bound = [right_dist + 0.5 * CAR_WIDTH for right_dist in boundary_right_dists]
    l_upper_bound = [left_dist - 0.5 * CAR_WIDTH for left_dist in boundary_left_dists]

    # dl bound
    dl_lower_bound = [-DL_LIMIT for i in range(ref_path_size)]
    dl_upper_bound = [DL_LIMIT for i in range(ref_path_size)]

    # ddl bound
    ddl_lower_bound = [-DDL_LIMIT for i in range(ref_path_size)]
    ddl_upper_bound = [DDL_LIMIT for i in range(ref_path_size)]

    # dddl bound
    dddl_lower_bound = [-DDDL_LIMIT for i in range(ref_path_size)]
    dddl_upper_bound = [DDDL_LIMIT for i in range(ref_path_size)]

    # contruct path optimizer and solve
    path_optimizer = PathOptimizer(len(ref_path_s_list))
    
    path_optimizer.SetCostingWeights(WEIGHT_L, WEIGHT_DL, WEIGHT_DDL, \
                                     WEIGHT_DDDL, WEIGHT_REF_L)
    path_optimizer.SetReferenceLList(ref_l_list)
    path_optimizer.SetStepList(ref_path_step_list)

    path_optimizer.SetInitState(init_state)
    path_optimizer.SetEndState(end_state)
    
    path_optimizer.SetLBound(l_upper_bound, l_lower_bound)
    path_optimizer.SetDlBound(dl_upper_bound, dl_lower_bound)
    path_optimizer.SetDdlBound(ddl_upper_bound, ddl_lower_bound)
    path_optimizer.SetDddlBound(dddl_upper_bound, dddl_lower_bound)

    # TODO：Complete this function
    path_optimizer.Solve()

    optimal_l, optimal_dl, optimal_ddl, optimal_dddl = path_optimizer.GetSolution()

    optimal_path_xs, optimal_path_ys, optimal_path_yaws, optimal_path_kappas, optimal_path_dkappas = \
            convert_state_from_frenet_to_cartesian(ref_path_x_list, ref_path_y_list, \
                ref_path_yaw_list, ref_path_kappa_list, optimal_l, optimal_dl, optimal_ddl, optimal_dddl)

    # plot results
    plt.figure(figsize=(12, 12))
    plt.axis("equal")
    plt.plot(ref_path_x_list, ref_path_y_list, "--k")
    
    plt.plot(optimal_path_xs, optimal_path_ys, '-', color='cornflowerblue')
    plot_path_range(optimal_path_xs, optimal_path_ys, optimal_path_yaws, width=2.0, color='royalblue')
    plt.plot(left_boundary_pt_xs, left_boundary_pt_ys, color='salmon', linewidth=2.0)
    plt.plot(right_boundary_pt_xs, right_boundary_pt_ys, color='salmon', linewidth=2.0)
    # plot_path_boxes(optimal_path_xs, optimal_path_ys, optimal_path_yaws, color='lightpink')
    plt.legend(["reference_path", "optimized_path", "path_range", "path boundary"])
    plt.title("Path Optimization")

    plt.figure(figsize=(12, 12))
    plt.subplot(4, 1, 1)
    plt.plot(ref_path_s_list, ref_l_list, '--', color='lightcoral')
    plt.plot(ref_path_s_list, optimal_l, color='cornflowerblue')
    plt.plot(ref_path_s_list, l_upper_bound, color='blueviolet')
    plt.plot(ref_path_s_list, l_lower_bound, color='blueviolet')
    plt.legend(["reference_l", "optimized_l", "l_bound"])

    plt.subplot(4, 1, 2)
    plt.plot(ref_path_s_list, optimal_dl, color='cornflowerblue')
    plt.plot(ref_path_s_list, dl_upper_bound, color='blueviolet')
    plt.plot(ref_path_s_list, dl_lower_bound, color='blueviolet')
    plt.legend(["optimized_dl", "dl_bound"])

    plt.subplot(4, 1, 3)
    plt.plot(ref_path_s_list, optimal_ddl, color='cornflowerblue')
    plt.plot(ref_path_s_list, ddl_upper_bound, color='blueviolet')
    plt.plot(ref_path_s_list, ddl_lower_bound, color='blueviolet')
    plt.legend(["optimized_ddl", "ddl_bound"])

    plt.subplot(4, 1, 4)
    plt.plot(ref_path_s_list, optimal_dddl, color='cornflowerblue')
    plt.plot(ref_path_s_list, dddl_upper_bound, color='blueviolet')
    plt.plot(ref_path_s_list, dddl_lower_bound, color='blueviolet')
    plt.legend(["optimized_dddl", "dddl_bound"])

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(ref_path_s_list, optimal_path_yaws, color='cornflowerblue')
    plt.legend(["theta"])
    plt.subplot(3, 1, 2)
    plt.plot(ref_path_s_list, optimal_path_kappas, color='cornflowerblue')
    plt.legend(["kappa"])
    plt.subplot(3, 1, 3)
    plt.plot(ref_path_s_list, optimal_path_dkappas, color='cornflowerblue')
    plt.legend(["dkappa"])

    plt.show()
    