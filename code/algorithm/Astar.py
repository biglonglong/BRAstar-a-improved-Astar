'''
Astar: Prioritise exploring points with minimum {explore_base+cost_heuristic}, dynamically plan explore_tree and explore_base, until find the goal
 - maintain consistency (cost_heuristic(s) <= cost_heuristic(s') + cost_neighbor(s,s')) to guarantee s' is optimal when s' heapqpoped)
'''

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + r"\..")

from map import Plot
from map import Env

import time

class astar:
    def __init__(self):
        self.env = Env.env()
        self.obs = self.env.obs
        
        self.source = self.env.source
        self.goal = self.env.goal

        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.open_set = []
        self.close_set = []
        self.explore_base = dict()
        self.explore_tree = dict()
    
    def cost_heuristic(self, point):
        if point in self.obs:
            return math.inf
        
        goal = self.goal
        x_dis = abs(goal[0] - point[0])
        y_dis = abs(goal[1] - point[1])

        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def cost_total(self, point):
        return self.explore_base[point] + self.cost_heuristic(point)

    def get_neighbor(self, point):
        return [(point[0] + move[0], point[1] + move[1]) for move in self.motions]

    def is_collision(self, start, end):
        if start in self.obs or end in self.obs:
            return True
        
        current_x,current_y,end_x,end_y = start[0],start[1],end[0],end[1]
        x_change = (end_x - current_x) / max(abs(end_x - current_x),1)
        y_change = (end_y - current_y) / max(abs(end_y - current_y),1)  

        while(current_x != end_x and current_y != end_y):
            current_x += x_change
            current_y += y_change
            if (current_x,current_y) in self.obs:
                return True

        while(current_x != end_x):
            current_x += x_change
            if (current_x,current_y) in self.obs:
                return True
        while(current_y != end_y):
            current_y += y_change
            if (current_x,current_y) in self.obs:
                return True
            
        return False
        
    def cost_neighbor(self, start, end):
        if self.is_collision(start, end):
            return math.inf
        
        x_dis = abs(end[0] - start[0])
        y_dis = abs(end[1] - start[1])

        return math.hypot(x_dis, y_dis)

    def extract_path(self):
        path = [self.goal]
        point_path = self.goal

        while True:
            point_path = self.explore_tree[point_path]
            path.append(point_path)
            
            if point_path == self.source:
                break

        return list(reversed(path))

    def count_corner(self, path):
        corners = 0
        for i in range(2, len(path)):
            if (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]) != (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1]):
                corners += 1

        return corners

    def calculate_safety(self, path):
        safty_count = 0
        for point in path:
            for motion in self.motions:
                if (point[0] + motion[0], point[1] + motion[1]) in self.obs:
                    safty_count += 1
                    break
        
        return safty_count / len(path)

    def torrent(self):
        for i in range(self.env.range_x):
            for j in range(self.env.range_y):
                self.explore_base[(i, j)] = math.inf
                self.explore_tree[(i, j)] = None

        self.explore_base[self.source] = 0
        self.explore_tree[self.source] = self.source
        heapq.heappush(self.open_set, (self.cost_total(self.source), self.source))

    def searching(self):
        self.torrent()

        while self.open_set:
            _, explore_point = heapq.heappop(self.open_set)
            self.close_set.append(explore_point)
            
            if explore_point == self.goal:  
                break

            for neighbor in self.get_neighbor(explore_point):
                new_cost = self.explore_base[explore_point] + self.cost_neighbor(explore_point, neighbor)

                if new_cost < self.explore_base[neighbor]:
                    self.explore_base[neighbor] = new_cost
                    self.explore_tree[neighbor] = explore_point
                    heapq.heappush(self.open_set, (self.cost_total(neighbor), neighbor))

        path = self.extract_path()
        return path, self.count_corner(path), self.calculate_safety(path), self.close_set

def main():
    AStar = astar()
    ploter = Plot.plot()

    T1 = time.time()
    path, corners, safety_factor, visited = AStar.searching()
    
    print('search time:\t', (time.time() - T1) * 2)
    print('search node:\t', len(visited))
    print('corner count:\t', corners)
    print('safty_factor:\t', safety_factor)
    print('path_length:\t', len(path))

    ploter.animation("Astar", path, False, "Astar", visited)
    
    plt.show()

if __name__ == '__main__':
    main()
