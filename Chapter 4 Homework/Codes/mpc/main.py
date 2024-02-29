import matplotlib.pyplot as plt
import math
import numpy as np

import sys
import pathlib
sys.path.append(str(pathlib.Path.cwd().parent.parent))

from mpc_controller import MPCTrajectoryTracker
from optimization_based_planner.frenet_path_optimizer import PathOptimizer
from utils.path_utils import *
from utils.vehicle_utils import plot_car_state, MAX_STEER, MAX_DSTEER, WHEEL_BASE
from utils.angle import path_yaws_normalization
from utils.angle import angle_mod
from utils.CubicSpline import cubic_spline_planner

N = 6 # mpc steps
DT = 0.2  # [s] mpc time resolution

CRUISING_SPEED = 8.0 / 3.6  # [m/s] target cruising speed
MAX_RUNNING_TIME = 500.0  # max simulation time

DIST_THRESHOLD = 1.5  # reach goal distance threshold
SPEED_THRESHOLD = 0.5 / 3.6  # reach goal speed threshold

REFRESH_FREQUENCY = 0.000001
SHOW_ANIMATION = True
SHOW_COMPLETE_RESULT = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def run_simulation(path_xs, path_ys, path_yaws, path_ks, speed_profile, ds, initial_state):
    """
    path_xs: path x position list
    path_ys: path y position list
    path_yaws: path yaw list
    path_ks: path curvature list
    speed_profile: speed profile
    ds: path step [m]
    initial_state: initial state of vehicle
    """
    path_yaws = path_yaws_normalization(path_yaws)

    goal = [path_xs[-1], path_ys[-1]]

    state = initial_state

    time = 0.0

    # initial yaw normalization
    state.yaw = angle_mod(state.yaw)

    # time list
    t_list = [0.0]

    # state lists
    x_list = [state.x]
    y_list = [state.y]
    yaw_list = [state.yaw]
    vel_list = [state.v]

    # input lists
    delta_list = [0.0]
    acc_list = [0.0]

    mpc_controller = MPCTrajectoryTracker(N, DT)

    nearest_idx, min_dist = get_nearest_index(
        state, path_xs, path_ys, path_yaws, 0)

    optimal_deltas, optimal_accs = None, None

    last_cycle_acc, last_cycle_delta = 0, 0
    while time <= MAX_RUNNING_TIME:
        ref_states, nearest_idx = get_ref_trajectory(
            state, path_xs, path_ys, path_yaws, path_ks, speed_profile, ds, nearest_idx)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        mpc_controller.update_initial_condition(
            state.x, state.y, state.yaw, state.v)

        mpc_controller.update_reference(
            ref_states[0, :], ref_states[1, :], ref_states[3, :], ref_states[2, :])

        mpc_controller.update_previous_input(last_cycle_acc, last_cycle_delta)

        sol_result = mpc_controller.solve()

        optimal_xs = sol_result['mpc_xs'][:, 0]
        optimal_ys = sol_result['mpc_xs'][:, 1]
        # only the first step of the control strategy is implemented.
        implemented_acc = sol_result['implemented_u'][0]
        implemented_delta = sol_result['implemented_u'][1]
        last_cycle_acc = implemented_acc
        last_cycle_delta = implemented_delta

        # update the vehicle state using mpc control
        state = update_vehicle_state(
            state, implemented_acc, implemented_delta, DT)

        time = time + DT
        t_list.append(time)
        # states
        x_list.append(state.x)
        y_list.append(state.y)
        yaw_list.append(state.yaw)
        vel_list.append(state.v)
        # inputs
        acc_list.append(implemented_acc)
        delta_list.append(implemented_delta)

        if check_arrival(state, goal, nearest_idx, len(path_xs)):
            print("Arrive Destination!")
            break

        if SHOW_ANIMATION:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plot_path_range(path_xs, path_ys, path_yaws,
                            width=2.0, color='royalblue')
            plt.plot(path_xs, path_ys, "-", color='salmon',
                     linewidth=2.0, label="reference path")
            plt.plot(x_list, y_list, "-", color='palegreen',
                     label="historical trajectory")
            plt.plot(ref_states[0, :], ref_states[1, :],
                     ".-k", label="reference states")
            if optimal_xs is not None and optimal_ys is not None:
                plt.plot(optimal_xs, optimal_ys, ".-",
                         color='blueviolet', label="MPC trajectory")
            # plt.plot(path_xs[nearest_idx], path_ys[nearest_idx], "xb", label="nearest index")
            plot_car_state(state.x, state.y, state.yaw,
                           steer=implemented_delta)
            plt.axis("equal")
            plt.title("MPC Controller")
            plt.legend()
            plt.pause(REFRESH_FREQUENCY)

    return t_list, x_list, y_list, yaw_list, vel_list, delta_list, acc_list


def generate_speed_profile(path_xs, path_ys, path_yaws, target_speed):
    speed_profile = [target_speed] * len(path_xs)

    direction = 1.0  # forward direction

    for i in range(len(path_xs) - 1):
        dx = path_xs[i + 1] - path_xs[i]
        dy = path_ys[i + 1] - path_ys[i]

        move_direction = math.atan2(dy, dx)

        # generate reversing speed if there is a significant difference
        # between the forward direction and the path direction.
        if dx != 0.0 and dy != 0.0:
            dangle = abs(angle_mod(move_direction - path_yaws[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        speed_profile[i] *= direction

    # Set stop point
    speed_profile[-1] = 0.0

    return speed_profile

# Given current state, return the index of the closest waypoint on the path.
def get_nearest_index(state, path_xs, path_ys, path_yaws, start_idx):
    N_IND_SEARCH = 10  # nearest index search steps
    dx_list = [
        state.x - path_x for path_x in path_xs[start_idx:(start_idx + N_IND_SEARCH)]]
    dy_list = [
        state.y - path_y for path_y in path_ys[start_idx:(start_idx + N_IND_SEARCH)]]

    comparable_dist_list = [dx ** 2 + dy **
                            2 for (dx, dy) in zip(dx_list, dy_list)]

    min_comp_dist = min(comparable_dist_list)

    min_dist_idx = comparable_dist_list.index(min_comp_dist) + start_idx

    min_dist = math.sqrt(min_comp_dist)

    x_diff = path_xs[min_dist_idx] - state.x
    y_diff = path_ys[min_dist_idx] - state.y

    # if current position is at the left of path, dist is defined as positive,
    # otherwise it is negative.
    angle_diff = angle_mod(
        path_yaws[min_dist_idx] - math.atan2(y_diff, x_diff))
    if angle_diff < 0:
        min_dist *= -1

    return min_dist_idx, min_dist

# Given current state, return the reference trajectory within the predicted horizon.
def get_ref_trajectory(state, path_xs, path_ys, path_yaws, path_ks, speed_profile, dl, start_idx):
    ref_states = np.zeros((4, N + 1))

    nearest_idx, _ = get_nearest_index(
        state, path_xs, path_ys, path_yaws, start_idx)

    # to prevent min dist index is behind start_idx.
    if nearest_idx <= start_idx:
        nearest_idx = start_idx

    ref_states[0, 0] = path_xs[nearest_idx]
    ref_states[1, 0] = path_ys[nearest_idx]
    ref_states[2, 0] = speed_profile[nearest_idx]
    ref_states[3, 0] = path_yaws[nearest_idx]

    point_nums = len(path_xs)
    accumulated_dist = 0.0  # accumulated distance of movement

    for i in range(N + 1):
        accumulated_dist += abs(state.v) * DT
        rel_idx = int(round(accumulated_dist / dl))

        if (nearest_idx + rel_idx) < point_nums:
            ref_states[0, i] = path_xs[nearest_idx + rel_idx]
            ref_states[1, i] = path_ys[nearest_idx + rel_idx]
            ref_states[2, i] = speed_profile[nearest_idx + rel_idx]
            ref_states[3, i] = path_yaws[nearest_idx + rel_idx]
        else:
            ref_states[0, i] = path_xs[point_nums - 1]
            ref_states[1, i] = path_ys[point_nums - 1]
            ref_states[2, i] = speed_profile[point_nums - 1]
            ref_states[3, i] = path_yaws[point_nums - 1]

    return ref_states, nearest_idx


def update_vehicle_state(state, a, delta, dt):
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.v = state.v + a * dt
    state.yaw = state.yaw + state.v / WHEEL_BASE * math.tan(delta) * dt
    return state


def check_arrival(state, goal, nearest_idx, total_idx):
    # check dist to goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    dist = math.hypot(dx, dy)

    reach_goal = (dist <= DIST_THRESHOLD)

    if abs(nearest_idx - total_idx) >= 5:
        reach_goal = False
    # check speed
    is_stop = (abs(state.v) <= SPEED_THRESHOLD)

    if reach_goal and is_stop:
        return True

    return False


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck


def generate_reference_line_and_boundary(ds=1.0):
    # generate reference line
    key_pt_xs = [10, 10, 20, 60, 65, 50, 40, 37, 40, 80]
    key_pt_ys = [10, 80, 80, 90, 45, 40, 70, 70, 20, 20]
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, ref_path_s_list = \
        generate_cublic_spline_reference_line(key_pt_xs, key_pt_ys, ds)
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


def run_path_optimization(ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list,
                          ref_path_s_list, boundary_left_dists, boundary_right_dists):

    ref_path_size = len(ref_path_s_list)
    ref_path_step_list = [ref_path_s_list[i+1] - ref_path_s_list[i]
                          for i in range(ref_path_size - 1)]
    ref_l_list = [0.5 * (boundary_right_dists[i] +
                         boundary_left_dists[i]) for i in range(ref_path_size)]

    init_state = [ref_l_list[0], 0.0, 0.0]
    end_state = [ref_l_list[-1], 0.0, 0.0]

    # state bounds
    l_lower_bound = [right_dist + 0.5 *
                     CAR_WIDTH for right_dist in boundary_right_dists]
    l_upper_bound = [left_dist - 0.5 *
                     CAR_WIDTH for left_dist in boundary_left_dists]

    # dl = (1 - kappa * l) * tan(delta_theta)
    dl_bound = math.tan(np.deg2rad(30))
    dl_lower_bound = [-dl_bound for i in range(ref_path_size)]
    dl_upper_bound = [dl_bound for i in range(ref_path_size)]

    # ddl = tan(max_delta)/wheel_base - k_r
    ddl_bound = (math.tan(MAX_STEER)/WHEEL_BASE - 0.0)
    ddl_lower_bound = [-ddl_bound for i in range(ref_path_size)]
    ddl_upper_bound = [ddl_bound for i in range(ref_path_size)]

    # dddl
    dddl_bound = MAX_DSTEER / WHEEL_BASE / 2.0
    dddl_lower_bound = [-dddl_bound for i in range(ref_path_size)]
    dddl_upper_bound = [dddl_bound for i in range(ref_path_size)]

    # contruct path optimizer and solve
    path_optimizer = PathOptimizer(len(ref_path_s_list))

    WEIGHT_L = 1.0
    WEIGHT_DL = 100.0
    WEIGHT_DDL = 1000.0
    WEIGHT_DDDL = 10000.0
    WEIGHT_REF_L = 1.0
    path_optimizer.SetCostingWeights(WEIGHT_L, WEIGHT_DL, WEIGHT_DDL,
                                     WEIGHT_DDDL, WEIGHT_REF_L)
    path_optimizer.SetReferenceLList(ref_l_list)
    path_optimizer.SetStepList(ref_path_step_list)

    path_optimizer.SetInitState(init_state)
    path_optimizer.SetEndState(end_state)

    path_optimizer.SetLBound(l_upper_bound, l_lower_bound)
    path_optimizer.SetDlBound(dl_upper_bound, dl_lower_bound)
    path_optimizer.SetDdlBound(ddl_upper_bound, ddl_lower_bound)
    path_optimizer.SetDddlBound(dddl_upper_bound, dddl_lower_bound)

    path_optimizer.Solve()

    optimal_l, optimal_dl, optimal_ddl, optimal_dddl = path_optimizer.GetSolution()

    optimal_path_xs, optimal_path_ys, optimal_path_yaws, optimal_path_kappas, optimal_path_dkappas = \
        convert_state_from_frenet_to_cartesian(ref_path_x_list, ref_path_y_list,
                                               ref_path_yaw_list, ref_path_kappa_list, optimal_l, optimal_dl, optimal_ddl, optimal_dddl)

    return optimal_path_xs, optimal_path_ys, optimal_path_yaws, optimal_path_kappas, optimal_path_dkappas


if __name__ == "__main__":
    # generate reference line and boundary
    ds = 1.0
    ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list, \
        ref_path_s_list, boundary_left_dists, boundary_right_dists = generate_reference_line_and_boundary(
            ds)

    left_boundary_pt_xs, left_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
        ref_path_x_list, ref_path_y_list, ref_path_yaw_list, boundary_left_dists)
    right_boundary_pt_xs, right_boundary_pt_ys = convert_pt_from_frenet_to_cartesian(
        ref_path_x_list, ref_path_y_list, ref_path_yaw_list, boundary_right_dists)

    optimal_path_xs, optimal_path_ys, optimal_path_yaws, optimal_path_kappas, optimal_path_dkappas = \
        run_path_optimization(ref_path_x_list, ref_path_y_list, ref_path_yaw_list, ref_path_kappa_list,
                              ref_path_s_list, boundary_left_dists, boundary_right_dists)

    # cx, cy, cyaw, ck = get_switch_back_course(ds)

    speed_profile = generate_speed_profile(
        optimal_path_xs, optimal_path_ys, optimal_path_yaws, CRUISING_SPEED)

    initial_state = State(
        x=optimal_path_xs[0], y=optimal_path_ys[0], yaw=optimal_path_yaws[0], v=0.0)

    plt.figure(figsize=(12, 12))

    # plt.axis("equal")
    t_list, x_list, y_list, yaw_list, vel_list, delta_list, acc_list = run_simulation(
        optimal_path_xs, optimal_path_ys, optimal_path_yaws, optimal_path_kappas, speed_profile, ds, initial_state)

    if SHOW_COMPLETE_RESULT:
        plt.close("all")
        plt.subplots(figsize=(12, 12))
        plt.axis("equal")
        plt.plot(optimal_path_xs, optimal_path_ys, "-",
                 color='cornflowerblue', label="refernece path")
        plt.plot(x_list, y_list, "-", color='lightpink',
                 label="tracking result")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()

        plt.subplots(figsize=(12, 12))
        plt.plot(t_list, vel_list, "-", color='cornflowerblue', label="speed")
        plt.xlabel("time (s)")
        plt.ylabel("velocity (km/h)")

        plt.show()
