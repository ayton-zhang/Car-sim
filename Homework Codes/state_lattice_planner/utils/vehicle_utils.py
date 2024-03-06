import math
import numpy as np

from matplotlib import pyplot as plt

# Vehicle geometry parameters
CAR_LENGTH = 4.5  # [m]
CAR_WIDTH = 2.0  # [m]
BACK_TO_REAR_AXLE = 1.0  # [m]
REAR_AXLE_TO_CENTER = CAR_LENGTH / 2.0 - BACK_TO_REAR_AXLE  # [m]

WHEEL_LENGTH = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WHEEL_BASE = 2.5  # [m]

# vehicle state limits
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -10.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

def plot_car_state(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):

    outline = np.array([[-BACK_TO_REAR_AXLE, (CAR_LENGTH - BACK_TO_REAR_AXLE), (CAR_LENGTH - BACK_TO_REAR_AXLE), -BACK_TO_REAR_AXLE, -BACK_TO_REAR_AXLE],
                        [CAR_WIDTH / 2, CAR_WIDTH / 2, - CAR_WIDTH / 2, -CAR_WIDTH / 2, CAR_WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LENGTH, -WHEEL_LENGTH, -WHEEL_LENGTH, WHEEL_LENGTH, WHEEL_LENGTH],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WHEEL_BASE
    fl_wheel[0, :] += WHEEL_BASE

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def plot_car_outline(x, y, yaw, color="-k"):

    outline = np.array([[-BACK_TO_REAR_AXLE, (CAR_LENGTH - BACK_TO_REAR_AXLE), (CAR_LENGTH - BACK_TO_REAR_AXLE), -BACK_TO_REAR_AXLE, -BACK_TO_REAR_AXLE],
                        [CAR_WIDTH / 2, CAR_WIDTH / 2, - CAR_WIDTH / 2, -CAR_WIDTH / 2, CAR_WIDTH / 2]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), color)