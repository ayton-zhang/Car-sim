import math
import numpy as np
from matplotlib import pyplot as plt
import sys
import pathlib
sys.path.append(str(pathlib.Path.cwd().parent))
from quintic_polynomial import QuinticPolynomial
from utils.angle import rot_mat_2d
from utils.path_utils import *

PATH_INTERVAL_M = 0.5
ROAD_HALF_WIDTH_M = 4.0
SAMPLE_STEP_M = 0.5
LON_SAFETY_DISTANCE_M = 1.0
LAT_SAFETY_DISTANCE_M = 0.5

WEIGHT_REFERENCE = 1.0
WEIGHT_SMOOTHNESS = 10.0
WEIGHT_SAFETY = 10000.0

OBSTACLES_X = [20.0, 23.0]
OBSTACLES_Y = [25.0, 24.0]

class Path:
    def __init__(self):
        self.s_list = []

        self.l_list = []
        self.dl_list = []
        self.ddl_list = []
        self.dddl_list = []

        self.x_list = []
        self.y_list = []
        self.yaw_list = []
        self.curvature_list = []


def generate_reference_line_and_boundary():
    # generate reference line
    key_pt_xs = [10, 20, 35]
    key_pt_ys = [10, 25, 30]
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, ref_path_s_list = \
            generate_cublic_spline_reference_line(key_pt_xs, key_pt_ys, ds=PATH_INTERVAL_M)

    boundary_left_dists = np.array([])
    boundary_right_dists = np.array([])
    for i in range(len(ref_path_s_list)):
        boundary_left_dists = np.append(
            boundary_left_dists, 0.0 + ROAD_HALF_WIDTH_M)
        boundary_right_dists = np.append(
            boundary_right_dists, 0.0 - ROAD_HALF_WIDTH_M)

    return ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
                 ref_path_s_list, boundary_left_dists, boundary_right_dists

def cal_path_cost(path):
    # TODO: calculate cost function: reference attraction cost, smoothness cost, safety cost, ...
    # return the total cost
    J_safe = WEIGHT_SAFETY * is_collision(path.x_list, path.y_list, path.yaw_list, OBSTACLES_X, OBSTACLES_Y)
    J_smooth = WEIGHT_SMOOTHNESS * sum([math.sqrt(dddl) for dddl in path.dddl_list])
    J_attractive = WEIGHT_REFERENCE * sum([s for s in path.s_list])
    return J_safe + J_smooth + J_attractive

def get_optimal_path(path_lattices_map):
    # TODO: return the minimum cost path
    path_key = min(path_lattices_map, key=lambda x: path_lattices_map[x])
    return path_key

def is_collision(x_list, y_list, yaw_list, obstacle_x_list, obstacle_y_list):
    # TODO: check if the given path(xs, ys, yaws) collides with obstacles
    pass

if __name__ == "__main__":
  
    # generate reference line and boundary
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
        ref_path_s_list, boundary_left_dists, boundary_right_dists = generate_reference_line_and_boundary()

    left_boundary_pt_xs, left_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
            ref_path_x_list,ref_path_y_list, ref_path_yaw_list, boundary_left_dists)
    right_boundary_pt_xs, right_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
            ref_path_x_list,ref_path_y_list, ref_path_yaw_list, boundary_right_dists)

    # initial state
    l_s = 0.0
    dl_s = 0.0
    ddl_s = 0.0

    # the key is path, and the value is cost of current path
    path_lattices_map = dict() 

    for l_e in np.arange(-ROAD_HALF_WIDTH_M, ROAD_HALF_WIDTH_M + SAMPLE_STEP_M, SAMPLE_STEP_M):
        path = Path()

        # TODO: 1. generate quintic polynomial path


        # TODO: 2. transform current path from frenet frame to cartesian frame.
        # refer to the function "convert_state_from_frenet_to_cartesian" in utils/path_utils.py

        
        # TODO: 3. costing each path
        path_lattices_map[path] = cal_path_cost(path)

    # TODO: 4. select the best path
    # extract optimal path
    optimal_path = get_optimal_path(path_lattices_map)

    # plot results
    plt.figure(figsize=(12, 12))
    plt.title("State Lattice Planning")
    plt.axis("equal")
    # plot reference path
    plt.plot(ref_path_x_list, ref_path_y_list, "--k")
    # plot obstacles
    plt.plot(OBSTACLES_X, OBSTACLES_Y, 'ok', markersize=10, zorder=100)
    # plot road range
    plt.plot(left_boundary_pt_xs, left_boundary_pt_ys, color='grey', linewidth=5.0, zorder=50)
    plt.plot(right_boundary_pt_xs, right_boundary_pt_ys, color='grey', linewidth=5.0, zorder=50)
    # plot optimal path
    plt.plot(optimal_path.x_list, optimal_path.y_list, '-', color='springgreen', zorder=30)
    # plot path lattices
    for path, cost in path_lattices_map.items():
        plt.plot(path.x_list, path.y_list, '-', color='cornflowerblue')
    # plot box coverage of optimal path
    plot_path_boxes(optimal_path.x_list, optimal_path.y_list, optimal_path.yaw_list, color='lightpink')

    plt.legend(["reference_path", "obstacles", "left road border", "right road border", "optimal path", "path lattices"])

    plt.show()
    