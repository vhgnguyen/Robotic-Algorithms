import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# vehicle 
W = 2.0 # [m]
LF = 3.7 # [m], distance from rear to front end
LB = 1.0 # [m], distance from rear to back end
LV = LF+LB # [m], vehicle length
TR = 0.5 # [m], tyre radius
TW = 1.3 # [m], tyre width
MAX_STEER = 0.6 # [rad], maximum steering angle
WB = 3 # [m], distance from rear to front steer

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W/2.0, -W/2.0, -W/2.0, W/2.0, W/2.0]

# collision check circles
WBUBBLE_R = (LF - LB)/2.0 + 0.5
WBUBBLE = (LF + LB) / 2.0 # distance from rear to bubble center


# rectangle collision check
def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = Rot.from_euler('z', yaw).as_dcm()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # no collision
    return True  # collision


def collision_check(x, y, yaw, ox, oy, kdtree):
    for (ix, iy, iyaw) in zip(x, y, yaw):
        cx = ix + WBUBBLE * math.cos(iyaw)
        cy = iy + WBUBBLE * math.sin(iyaw)

        # check the whole bubble
        idx = kdtree.query_ball_point([cx, cy], WBUBBLE_R)
        if len(idx) == 0:
            continue
        tmp_ox, tmp_oy = [], []
        for i in idx:
            tmp_ox.append(ox[i])
            tmp_oy.append(oy[i])
        if not rectangle_check(ix, iy, iyaw, tmp_ox, tmp_oy):
            return False
    return True


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def updateState(x, y, yaw, direction, steer, L=WB):
    x += direction * math.cos(yaw)
    y += direction * math.sin(yaw)
    yaw += pi_2_pi(direction * math.tan(steer) / L)
    return x, y, yaw


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


# plot vehicle
def plot_vehicle(x, y, yaw, steer):
    vehicleColor = "-k"
    vehicleOutline = np.array([[-LB, LF, LF, -LB, -LB], 
                               [W/2, W/2, -W/2, -W/2, W/2]])
    rrWheel = np.array([[TR, -TR, -TR, TR, TR],
                        [-W/12.0+TW,-W/12.0+TW, W/12.0+TW, W/12.0+TW, -W/12.0+TW]])

    rlWheel = np.array([[TR, -TR, -TR, TR, TR],
                        [-W/12.0-TW, -W/12.0-TW, W/12.0-TW, W/12.0-TW, -W/12.0-TW]])

    frWheel = np.array([[TR, -TR, -TR, TR, TR],
                        [-W/12.0+TW, -W/12.0+TW, W/12.0+TW, W/12.0+TW, -W/12.0+TW]])

    flWheel = np.array([[TR, -TR, -TR, TR, TR],
                        [-W/12.0-TW, -W/12.0-TW, W/12.0-TW, W/12.0-TW, -W/12.0-TW]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                    [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])
    
    frWheel = np.dot(frWheel.T, Rot2).T
    flWheel = np.dot(flWheel.T, Rot2).T
    frWheel[0, :] += WB
    flWheel[0, :] += WB
    frWheel = np.dot(frWheel.T, Rot1).T
    flWheel = np.dot(flWheel.T, Rot1).T

    vehicleOutline = np.dot(vehicleOutline.T, Rot1).T

    rrWheel = np.dot(rrWheel.T, Rot1).T
    rlWheel = np.dot(rlWheel.T, Rot1).T

    vehicleOutline[0, :] += x
    vehicleOutline[1, :] += y
    flWheel[0, :] += x
    flWheel[1, :] += y
    frWheel[0, :] += x
    frWheel[1, :] += y
    rlWheel[0, :] += x
    rlWheel[1, :] += y
    rrWheel[0, :] += x
    rrWheel[1, :] += y

    c, s = math.cos(yaw), math.sin(yaw)
    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(x, y, "*")
    plt.plot(frWheel[0, :], frWheel[1, :], vehicleColor)
    plt.plot(rrWheel[0, :], rrWheel[1, :], vehicleColor)
    plt.plot(flWheel[0, :], flWheel[1, :], vehicleColor)
    plt.plot(rlWheel[0, :], rlWheel[1, :], vehicleColor)
    plt.plot(vehicleOutline[0, :], vehicleOutline[1, :], vehicleColor)


def main():
    x, y, yaw, steer = 0., 0., 1., 0.3
    print("Plot vehicle state [x, y, yaw, steer]: [%.2f, %.2f, %.2f, %.2f]" %(x, y, yaw, steer))
    plt.axis('equal')
    plot_vehicle(x, y, yaw, steer)
    plt.show()


if __name__ == '__main__':
    main()