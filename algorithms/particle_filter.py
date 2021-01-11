# -*- coding: utf-8 -*-
# @Author: Vo Hoang Nguyen
# @Date:   2021-01-11 14:42:04
# @Last Modified by:   Vo Hoang Nguyen
# @Last Modified time: 2021-01-11 14:42:10

import math
import matplotlib.pyplot as plt
import numpy as np

## Parameter
# PF estimation error 
sigma_d = 1
Qp = np.diag([sigma_d]) ** 2 # distance error
Rp = np.diag([2, np.deg2rad(40)]) ** 2 # input error

# Observation error
Qobs = np.diag([0.2]) ** 2 # distance error
Robs = np.diag([2, np.deg2rad(40)]) ** 2 # input error

# Simulation parameters
SIMU_TIME = 50 # simulation time [s]
dT = 0.1 # time stick [s]
RANGE = 25 # observation range [m]

# PF parameters
nP = 1000 # number of particle
nRP = nP / 2.0 # number of resampling particle

# System matrix
A = np.array([[1.0, 0, 0, 0],
              [0, 1.0, 0, 0],
              [0, 0, 1.0, 0],
              [0, 0, 0, 0]])

# Input matrix
def B(x):
    return  np.array([[dT * math.cos(x[2, 0]), 0],
                      [dT * math.sin(x[2, 0]), 0],
                      [0.0, dT],
                      [1.0, 0.0]])


# Nonlinear two wheel motion model
def updateState(x, u):
    return A.dot(x) + B(x).dot(u)


# Bivariante gaussian
def gaussian(x, sigma):
    return 1.0 / math.sqrt(2.0 * math.pi * sigma**2) * \
        math.exp(-x**2 / (2 * sigma**2))


def covariance(x_es, px, pw):
    """
    Param:
        x_es: estimated state
        px, pw: particle and weight
    """
    cov = np.zeros((3,3))
    nP = px.shape[1]
    for i in range(nP):
        dx = (px[:, i:i+1] - x_es)[0:3]
        cov += pw[0, i] * dx @ dx.T
    return  cov * 1.0 / (1.0 - pw@pw.T)


def observation(x_gt, x_d, u, lm):
    """
    Update observation model
    Param:
        x_gt: ground truth states
        x_o: observed states from motion model
        u: control input
        lm: landmark
    """
    # get a state
    un_gt = np.array([[u[0, 0] + np.random.randn() * Robs[0, 0] ** 0.5 / 10,
                       u[1, 0] + np.random.randn() * Robs[1, 1] ** 0.5 / 10]]).T
    x_gt = updateState(x_gt, un_gt)

    # landmark observation list
    zn = np.zeros((0, 3))

    for i in range(len(lm[:, 0])):
        # calculate the distance from current pose to all landmarks
        dx = x_gt[0, 0] - lm[i, 0]
        dy = x_gt[1, 0] - lm[i, 1]
        d = math.hypot(dx, dy)

        if d <= RANGE:
            # add noise and add the distance to observation list
            d += np.random.randn() * Qobs[0, 0] ** 0.5
            zn = np.vstack((zn, np.array([[d, lm[i, 0], lm[i, 1]]])))

    # add noise to dead reckoning input
    un = np.array([[u[0, 0] + np.random.randn() * Robs[0, 0] ** 0.5,
                    u[1, 0] + np.random.randn() * Robs[1, 1] ** 0.5]]).T
    
    # update dead reckoning motion model
    x_d = updateState(x_d, un)

    return x_gt, zn, x_d, un


def pf_localization(px, pw, zn, u):
    """
    Particle filter localization
    Param:
        px: particle pose
        pw: particle weight
        zn: observed landmarks with noise
        u: control input
    """
    for ip in range(px.shape[1]):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]

        # Predict particle state
        upn = np.array([[u[0, 0] + np.random.randn() * Rp[0, 0] ** 0.5,
                         u[1, 0] + np.random.randn() * Rp[1, 1] ** 0.5]]).T

        # Update particle
        x = updateState(x, upn)

        # calculate weight from observed landmarks
        for i in range(len(zn[:, 0])):
            dx = x[0, 0] - zn[i, 1]
            dy = x[1, 0] - zn[i, 2]
            pz = math.hypot(dx, dy)
            dz = pz - zn[i, 0]
            w = w * gaussian(dz, sigma_d)
        
        # update particle
        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    # normalize the weight
    pw = pw / pw.sum()

    # update estimated state with covariance
    x_es = px.dot(pw.T)
    cov_es = covariance(x_es, px, pw)

    # resampling
    nEff = 1. / (pw.dot(pw.T))[0,0]
    if nEff < nRP:
        px, pw = re_sampling(px, pw)
    
    return x_es, cov_es, px, pw


def re_sampling(px, pw):
    """ 
    Resampling particle
    """
    w_cum = np.cumsum(pw)
    base = np.arange(0., 1., 1 / nP) + np.random.uniform(0, 1 / nP)
    idx = []
    count = 0
    for ip in range(nP):
        while base[ip] > w_cum[count]:
            count += 1
        idx.append(count)
    
    # resampling weight
    pw = np.zeros((1, nP)) + 1. / nP
    
    return px[:, idx], pw


def main():
    print(__file__ + " start!!")

    time = 0.0

    # lm positions [x, y]
    lm = np.array([[10.0, 0.0],
                  [10.0, 12.0],
                  [0.0, 25.0],
                  [15.0, 20.0],
                  [25.0, 10.0]])

    # State Vector [x y yaw v]'
    x_es = np.zeros((4, 1))
    x_gt = np.zeros((4, 1))

    px = np.zeros((4, nP))  # Particle store
    pw = np.zeros((1, nP)) + 1.0 / nP  # Particle weight
    x_d = np.zeros((4, 1))  # Dead reckoning

    # history
    h_x_es = x_es
    h_x_gt = x_gt
    h_x_d = x_gt

    while SIMU_TIME >= time:
        time += dT
        u = np.array([[2.0, 0.15]]).T

        x_gt, z, x_d, ud = observation(x_gt, x_d, u, lm)

        x_es, cov_es, px, pw = pf_localization(px, pw, z, ud)

        # store data history
        h_x_es = np.hstack((h_x_es, x_es))
        h_x_d = np.hstack((h_x_d, x_d))
        h_x_gt = np.hstack((h_x_gt, x_gt))

        if True:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            for i in range(len(z[:, 0])):
                plt.plot([x_gt[0, 0], z[i, 1]], [x_gt[1, 0], z[i, 2]], ls=":", c="g" , alpha=0.5)
            plt.scatter(lm[:, 0], lm[:, 1], c='g', linewidths=5, label="landmark")
            plt.scatter(px[0, :], px[1, :], c="lightcoral", s=1, linewidths=2, alpha=0.5, label="particle")
            plt.plot(np.array(h_x_es[0, :]).flatten(),
                     np.array(h_x_es[1, :]).flatten(), "-r", lw=2, alpha=1, label="pf estimated state")
            plt.plot(np.array(h_x_gt[0, :]).flatten(),
                     np.array(h_x_gt[1, :]).flatten(), "-.b", lw=1, alpha=0.7, label="ground truth state")
            plt.plot(np.array(h_x_d[0, :]).flatten(),
                     np.array(h_x_d[1, :]).flatten(), "-.k", lw=1, alpha=0.7, label="dead reckoning state")
            plt.axis("equal")
            plt.legend()
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()



        
