import math
import numpy as np
import trajectory_generation as planner
import bicycle_motion_model as bmm
import pandas as pd
import matplotlib.pyplot as plt


def calc_states_list():
    maxyaw = np.deg2rad(-30.0)
    x = np.arange(10.0, 30.0, 5.0)
    y = np.arange(0.0, 20.0, 2.0)
    yaw = np.arange(-maxyaw, maxyaw, maxyaw)
    states = []
    for iyaw in yaw:
        for iy in y:
            for ix in x:
                states.append([ix, iy, iyaw])
    return states


def search_nearest_one_from_lookuptable(tx, ty, tyaw, lookuptable):
    mind = float("inf")
    minid = -1
    for (i, table) in enumerate(lookuptable):
        dx = tx - table[0]
        dy = ty - table[1]
        dyaw = tyaw - table[2]
        d = math.sqrt(dx**2 + dy**2 + dyaw**2)
        if d <= mind:
            minid = i 
            mind = d
    
    return lookuptable[minid]


def save_lookup_table(fname, table):
    mt = np.array(table)
    df = pd.DataFrame()
    df["x"] = mt[:, 0]
    df["y"] = mt[:, 1]
    df["yaw"] = mt[:, 2]
    df["s"] = mt[:, 3]
    df["km"] = mt[:, 4]
    df["kf"] = mt[:, 5]
    df.to_csv(fname, index=None)


def generate_lookup_table():
    states = calc_states_list()
    k0 = 0.0
    lookuptable = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    for state in states:
        bestP = search_nearest_one_from_lookuptable(
            state[0], state[1], state[2], lookuptable)
        target = bmm.State(x=state[0], y=state[1], yaw=state[2])
        initP = np.array(
            [math.sqrt(state[0]**2 + state[1]**2), bestP[4], bestP[5]]).reshape(3,1)
        x, y, yaw, p = planner.optimize_trajectory(target, k0, initP)
        if x is not None:
            print("find good path")
            lookuptable.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])
    save_lookup_table("lookup_table.csv", lookuptable)
    for table in lookuptable:
        xc, yc, yawc = bmm.generate_trajectory(
            table[3], table[4], table[5], k0)
        plt.plot(xc, yc, "-r")
        xc, yc, yawc = bmm.generate_trajectory(
            table[3], -table[4], -table[5], k0)
        plt.plot(xc, yc, "-r")

    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    generate_lookup_table()


if __name__ == "__main__":
    main()