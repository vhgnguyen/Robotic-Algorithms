# -*- coding: utf-8 -*-
# @Author: Vo Hoang Nguyen
# @Date:   2021-01-13 17:46:43
# @Last Modified by:   Vo Hoang Nguyen
# @Last Modified time: 2021-01-13 21:57:48
import math
import matplotlib.pyplot as plt
import numpy as np
import heapq

from scipy.spatial import cKDTree
from sklearn import neighbors

import reeds_shepp_path as rs
from a_star import calc_distance_heuristic
from vehicle import collision_check, updateState, plot_vehicle, MAX_STEER, WB

# constant
XY_GRID_RES = 2.0 # [m]
YAW_GRID_RES = np.deg2rad(15.0) # [rad]
PATH_RES = 0.1 # [m] interpolation resolution
N_STEER = 20 # number of steer command

# cost weight
SB_COST = 100.0 # switch back cost
BACK_COST = 5.0 # backward cost
STEER_CHANGE_COST = 5.0 # steer angle change cost
STEER_COST = 1.0 # steering angle cost
H_COST = 5.0 # heuristic cost

# plot animation
showAnimation = True


class Node:
    def __init__(self, xind, yind, yawind, direction, xlist, ylist, yawlist, directions, cost, steer=0.0, pind=0):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Path:
    def __init__(self, xlist, ylist, yawlist, directionlist, cost):
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = directionlist
        self.cost = cost


class Config:
    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)
        # ox.append(min_x_m)
        # oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.minx = round(min_x_m / xy_resolution)
        self.maxx = round(max_x_m / xy_resolution)
        self.miny =  round(min_y_m / xy_resolution)
        self.maxy = round(max_y_m / xy_resolution)

        self.xw = round(self.maxx - self.minx)
        self.yw = round(self.maxy - self.miny)

        self.minyaw = round(- math.pi / yaw_resolution) - 1
        self.maxyaw = round(math.pi / yaw_resolution)
        self.yaww = round(self.maxyaw - self.minyaw)


class HybridAStar:

    def __init__(self, start, goal, ox, oy, xy_res, yaw_res):
        start[2] = rs.pi_2_pi(start[2])
        goal[2] = rs.pi_2_pi(goal[2])
        tox = ox[:]
        toy = oy[:]
        self.kdtree = cKDTree(np.vstack((tox, toy)).T)
        self.config = Config(ox, oy, xy_res, yaw_res)
        self.startNode = Node(round(start[0] / xy_res),
                              round(start[1] / xy_res),
                              round(start[2] / yaw_res), True,
                              [start[0]], [start[1]], [start[2]], [True], cost=0)
        self.goalNode = Node(round(goal[0] / xy_res),
                             round(goal[1] / xy_res),
                             round(goal[2] / yaw_res), True,
                             [goal[0]], [goal[1]], [goal[2]], [True], cost=0)
        
        # planning
        openList, closedList = {}, {}
        p_Astar = calc_distance_heuristic(
            self.goalNode.xlist[-1], self.goalNode.ylist[-1], ox, oy, xy_res, rr=1)
        pq = []
        openList[getIndex(self.startNode, self.config)] = self.startNode
        heapq.heappush(pq, (getHeuristicCost(self.startNode, p_Astar, self.config),
                       getIndex(self.startNode, self.config)))
        finalPath = None

        while True:
            if not openList:
                print("Error: Cannot find path, No open set")
                return [], [], []

            cost, c_id = heapq.heappop(pq)
            if c_id in openList:
                current = openList.pop(c_id)
                closedList[c_id] = current
            else:
                continue

            if showAnimation:
                plt.plot(current.xlist[-1], current.ylist[-1], "xc")
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closedList.keys()) % 10 == 0:
                    plt.pause(0.001)

            is_updated, finalPath = updateNodeWithExpansion(
                current, self.goalNode, ox, oy, self.kdtree, self.config)

            if is_updated:
                print("Path found")
                break

            for neighbor in getNeighbor(current, ox, oy, self.kdtree, self.config):
                nind = getIndex(neighbor, self.config)
                if nind in closedList:
                    continue
                if neighbor not in openList or openList[nind].cost > neighbor.cost:
                    heapq.heappush(pq, (getHeuristicCost(neighbor, p_Astar, self.config), nind))
                    openList[nind] = neighbor

        self.path = getFinalPath(closedList, finalPath)


def motionInput():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for direction in [1, -1]:
            yield [steer, direction]


def getNeighbor(cnode, ox, oy, kdtree, config):
    for steer, direction in motionInput():
        node = getNextNode(cnode, steer, direction, ox, oy, kdtree, config)
        if node and verifyIndex(node, config):
            yield node


def getNextNode(cnode, steer, direction, ox, oy, kdtree, config):
    x, y, yaw = cnode.xlist[-1], cnode.ylist[-1], cnode.yawlist[-1]

    arcLength = XY_GRID_RES * 1.5
    xlist, ylist, yawlist = [], [], []
    for _ in np.arange(0, arcLength, PATH_RES):
        x, y, yaw = updateState(x, y, yaw, direction, steer)
        xlist.append(x)
        ylist.append(y)
        yawlist.append(yaw)
    
    if not collision_check(xlist, ylist, yawlist, ox, oy, kdtree):
        return None
    
    xind = round(x / XY_GRID_RES)
    yind = round(y / XY_GRID_RES)
    yawind = round(yaw / YAW_GRID_RES)

    cost = 0.0

    # direction change cost
    if direction != cnode.direction:
        cost += SB_COST
    # steer cost
    cost += STEER_COST * abs(steer)
    # steer change cost
    cost += STEER_CHANGE_COST * abs(cnode.steer - steer)
    cost += cnode.cost + arcLength
    
    # new node
    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, [direction],
                steer=steer, cost=cost, pind=getIndex(cnode, config))
    return node


def checkSameGrid(node1, node2):
    if node1.xind != node2.xind or node1.yind != node2.yind or \
        node1.yawind != node2.yawind:
        return False
    return True


# reeds shepp path cost
def rsPathCost(rsPath):
    cost = 0.0
    # forward-backward cost
    for l in rsPath.lengths:
        if l >= 0:
            cost += l
        else:
            cost += abs(l) * BACK_COST
    # switch back cost
    for i in range(len(rsPath.lengths) - 1):
        if rsPath.lengths[i] * rsPath.lengths[i+1] < 0.0:
            cost += SB_COST
    # steer cost
    for c_type in rsPath.ctypes:
        if c_type != "S":
            cost += STEER_COST * abs(MAX_STEER)
    # steer change cost
    n_steer = len(rsPath.ctypes)
    u = [0.0] * n_steer
    for i in range(n_steer):
        if rsPath.ctypes[i] == "R":
            u[i] = - MAX_STEER
        elif rsPath.ctypes[i] == "L":
            u[i] = MAX_STEER
    for i in range(n_steer-1):
        cost += STEER_CHANGE_COST * abs(u[i+1] - u[i])
    return cost


def nodeExpansion(cnode, goal, ox, oy, kdtree):
    startx, starty, startyaw = cnode.xlist[-1], cnode.ylist[-1], cnode.yawlist[-1]
    goalx, goaly, goalyaw = goal.xlist[-1], goal.ylist[-1], goal.yawlist[-1]

    maxCurvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(startx, starty, startyaw, goalx, goaly, goalyaw, maxCurvature,PATH_RES)
    if not paths:
        return None
    bestPath, bestCost = None, None
    for path in paths:
        if collision_check(path.x, path.y, path.yaw, ox, oy, kdtree):
            cost = rsPathCost(path)
            if not bestCost or bestCost > cost:
                bestCost = cost
                bestPath = path
    return bestPath


# update forward node expansion
def updateNodeWithExpansion(cnode, goal, ox, oy, kdtree, config):
    path = nodeExpansion(cnode, goal, ox, oy, kdtree)
    if path:
        if showAnimation:
            plt.plot(path.x, path.y, "xc", ms=5)
            plt.waitforbuttonpress(0)
        x = path.x[1:]
        y = path.y[1:]
        yaw = path.yaw[1:]
        cost = cnode.cost + rsPathCost(path)
        pind = getIndex(cnode, config)
        directions = []
        for d in path.directions[1:]:
            directions.append(d>=0)
        steer = 0.0
        fpath = Node(cnode.xind, cnode.yind, cnode.yawind, cnode.direction,
                     x, y, yaw, directions, steer,cost, pind)
        return True, fpath
    return False, None


def getIndex(node, config):
    ind = (node.yawind - config.minyaw)*config.xw*config.yw + \
          (node.yind - config.miny)*config.xw + (node.xind - config.minx)
    if ind < 0:
        print("Error when getting index: ", ind)
    return ind


def verifyIndex(node, config):
    xind, yind = node.xind, node.yind
    if config.minx <= xind and config.maxx >= xind and \
        config.miny <= yind and config.maxy >= yind:
        return True
    return False
    

def getHeuristicCost(node, p_Astar, config):
    # A-star index
    ind = (node.yind - config.miny) * config.xw + (node.xind - config.minx)
    if ind not in p_Astar:
        return math.inf
    return node.cost + H_COST * p_Astar[ind].cost


def getFinalPath(closed, goalNode):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goalNode.xlist)), list(reversed(goalNode.ylist)), \
        list(reversed(goalNode.yawlist))
    direction = list(reversed(goalNode.directions))
    nid = goalNode.pind
    final_cost = goalNode.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.xlist)))
        reversed_y.extend(list(reversed(n.ylist)))
        reversed_yaw.extend(list(reversed(n.yawlist)))
        direction.extend(list(reversed(n.directions)))
        nid = n.pind

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def main():

    print("Start Hybrid A* planning")

    ox, oy = [], []

    for i in range(-25, 26):
        ox.append(i)
        oy.append(17.0)
    for i in range(-25, -1):
        ox.append(i)
        oy.append(4.0)
    for i in range(-1, 5):
        ox.append(-8.0)
        oy.append(i)
    for i in range(-1, 5):
        ox.append(8.0)
        oy.append(i)
    for i in range(8, 25):
        ox.append(i)
        oy.append(4.0)
    for i in range(-8, 8):
        ox.append(i)
        oy.append(-1.0)
    for i in range(4, 17):
        ox.append(-25)
        oy.append(i)
    for i in range(4, 17):
        ox.append(25)
        oy.append(i)
    for i in range(4, 9):
        ox.append(-8)
        oy.append(i)

    # Set Initial parameters
    start = [-20.0, 9.0, np.deg2rad(0.0)]
    goal = [-2.0, 1.0, np.deg2rad(0.0)]
    print("Start : ", start)
    print("Goal : ", goal)

    if showAnimation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')
        rs.plot_arrow(goal[0], goal[1], goal[2], fc='g')
        plt.grid(True)
        plt.axis("equal")
        plt.waitforbuttonpress(0)

    hybridAStar = HybridAStar(start, goal, ox, oy, XY_GRID_RES, YAW_GRID_RES)
    path = hybridAStar.path
    x = path.xlist
    y = path.ylist
    yaw = path.yawlist

    if showAnimation:
        for i in range(len(x)-1):
            i_x = x[i]
            i_y = y[i]
            i_yaw = yaw[i]
            plt.cla()
            plt.plot(ox, oy, ".b", label="Obstacles")
            plt.plot(x, y, "-r", label="Hybrid A* path")
            plot_vehicle(i_x, i_y, i_yaw, yaw[i+1]-yaw[i])
            plt.grid(True)
            plt.axis("equal")
            plt.legend()
            plt.pause(0.0001)

    print("Done!!")
    plt.waitforbuttonpress(0)
    plt.close(fig)

if __name__ == '__main__':
    main()