import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import bicycle_motion_model as bmm
import trajectory_generation as planner

show_animation = True

table_path = "lookup_table.csv"


def search_nearest_from_lookuptable(tx, ty, tyaw, table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(table):
        dx = tx - table[0]
        dy = ty - table[1]
        dyaw = tyaw - table[2]
        d = math.sqrt(dx**2 + dy**2 + dyaw**2)
        if d <= mind:
            minid = i
            mind = d
    return table[minid]


def get_lookuptable():
    data = pd.read_csv(table_path)
    return np.array(data)


def generate_path(states, k0):
    table = get_lookuptable()
    result = []

    for state in states:
        bestP = search_nearest_from_lookuptable(
            state[0], state[1], state[2], table)
        target = bmm.State(x=state[0], y=state[1], yaw=state[2])
        initP = np.array([math.sqrt(state[0]**2 + state[1]**2), bestP[4], bestP[5]]).reshape(3,1)
        x, y, yaw, p = planner.optimiz_trajectory(target, k0, initP)
        if x is not None:
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])
    return result


def calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max):
    angle_samples = [i / (nxy - 1) for i in range(nxy)]
    states = samples_states(angle_samples, a_min, a_max, d, p_max, p_min, nh)
    return states


def sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh):
    states = []
    for i in angle_samples:
        a = a_min + (a_max, a_min) * i
        for j in range(nh):
            xf = d * math.cos(a)
            yf = d * math.sin(a)
            if nh == 1:
                yawf = (p_max - p_min) / 2 + a
            else:
                yawf = p_min + (p_max - p_min) * 