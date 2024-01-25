import math
import numpy as np

from matplotlib import pyplot as plt
from utils.angle import angle_mod
from utils.CubicSpline import cubic_spline_planner
from utils.vehicle_utils import plot_car_outline

# vehicle parameters for plotting
CAR_LENGTH = 4.5  # [m]
CAR_WIDTH = 2.0  # [m]
BACK_TO_REAR_AXLE = 1.0  # [m]
REAR_AXLE_TO_CENTER = CAR_LENGTH / 2.0 - BACK_TO_REAR_AXLE  # [m]

def generate_cublic_spline_reference_line(key_pt_xs, key_pt_ys, ds=1.0):
    x_list, y_list, yaw_list, kappa_list, s_list = cubic_spline_planner.calc_spline_course(
        key_pt_xs, key_pt_ys, ds=ds)
    return x_list, y_list, yaw_list, kappa_list, s_list


def convert_pt_from_frenet_to_cartesian(ref_path_xs, ref_path_ys, ref_path_yaws, ls):
    x_list = []
    y_list = []

    for ref_x, ref_y, ref_yaw, l in zip(ref_path_xs, ref_path_ys, ref_path_yaws, ls):
        x_list.append(ref_x - math.sin(ref_yaw) * l)
        y_list.append(ref_y + math.cos(ref_yaw) * l)

    return x_list, y_list

def convert_state_from_frenet_to_cartesian(ref_path_xs, ref_path_ys, ref_path_yaws, ref_path_kappas, \
                                            ls, dls, ddls, dddls):
    x_list, y_list = convert_pt_from_frenet_to_cartesian(ref_path_xs, ref_path_ys, ref_path_yaws, ls)
    yaw_list = []
    for ref_yaw, ref_kappa, l, dl in zip(ref_path_yaws, ref_path_kappas, ls, dls):
        yaw_list.append(math.atan(dl / (1 - ref_kappa * l)) + ref_yaw)

    kappa_list = []
    # the curvature of the optimal path is approximated by numerical difference here
    for idx in range(len(yaw_list) - 1):
        dx = x_list[idx + 1] - x_list[idx]
        dy = y_list[idx + 1] - y_list[idx]
        ds = math.sqrt(dx * dx + dy * dy)
        kappa_list.append((angle_mod(yaw_list[idx + 1] - yaw_list[idx])) / ds)
    kappa_list.append(kappa_list[-1])
    
    dkappa_list = []
    for idx in range(len(kappa_list) - 1):
        dx = x_list[idx + 1] - x_list[idx]
        dy = y_list[idx + 1] - y_list[idx]
        ds = math.sqrt(dx * dx + dy * dy)
        dkappa_list.append((kappa_list[idx + 1] - kappa_list[idx]) / ds)
    dkappa_list.append(dkappa_list[-1])

    return x_list, y_list, yaw_list, kappa_list, dkappa_list

def plot_path_boxes(path_xs, path_ys, path_yaws, color='k'):
    count = 0
    for x, y, yaw in zip(path_xs, path_ys, path_yaws):
        if (count % 2 == 0):
            plot_car_outline(x, y, yaw, color)
        count += 1

def plot_path_range(x_list, y_list, yaw_list, width, color):
    path_left_dists = [0.5 * width] * len(x_list)
    path_right_dists = [-0.5 * width] * len(x_list)

    path_left_x, path_left_y = convert_pt_from_frenet_to_cartesian(x_list, \
                    y_list, yaw_list, path_left_dists)
    path_right_x, path_right_y = convert_pt_from_frenet_to_cartesian(x_list, \
                    y_list, yaw_list, path_right_dists)

    plt.fill(np.concatenate([path_left_x, path_right_x[::-1]]),\
             np.concatenate([path_left_y, path_right_y[::-1]]), alpha=0.3, color=color)