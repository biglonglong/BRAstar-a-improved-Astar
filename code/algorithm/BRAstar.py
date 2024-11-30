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

class brastar:
    def __init__(self):
        self.env = Env.env()
        self.obs = self.env.obs
        
        self.source = self.env.source
        self.goal = self.env.goal

        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.open_set_for = []
        self.open_set_back = []
        self.close_set_for = []
        self.close_set_back = []
        self.explore_base_for = dict()
        self.explore_base_back = dict()
        self.explore_tree_for = dict()
        self.explore_tree_back = dict()

        self.incons_set_for = [] 
        self.incons_set_back = [] 
        self.visited_for = []
        self.visited_back = []

        self.point_meet = self.source
        self.cost_total_point_meet = math.inf

        self.costmap = self.env.map
        self.expansion_radius = 1
        self.alpha = 0.25
        
        self.parent_direction = dict()
        self.corner_loss = 1.0

    def costmap_expansion(self, expansion_unit, obs_point):
        if expansion_unit > self.expansion_radius: 
            return
        else:
            for motion in self.motions:
                swell_x, swell_y = (motion[0] + obs_point[0], motion[1] + obs_point[1])
                if 0 <= swell_x < self.env.range_x and 0 <= swell_y < self.env.range_y:
                    self.costmap[swell_x][swell_y] = max(self.costmap[swell_x][swell_y], 1.0 / np.sqrt(expansion_unit+1))
                self.costmap_expansion(expansion_unit + 1, (swell_x, swell_y))

    def update_costmap(self):
        for obs_point in self.obs:
            self.costmap_expansion(1, obs_point)

    def cost_heuristic(self, point, goal):
        if point in self.obs:
            return math.inf
        
        x_dis = abs(goal[0] - point[0])
        y_dis = abs(goal[1] - point[1])

        return x_dis + y_dis - 0.586 * min(x_dis, y_dis)
    
    def cost_total_for(self, point):
        return self.explore_base_for[point] + (self.alpha + (1.0 - self.alpha) * (1.0 + self.costmap[point[0]][point[1]])) * self.cost_heuristic(point, self.goal)

    def cost_total_back(self, point):
        return self.explore_base_back[point] + (self.alpha + (1.0 - self.alpha) * (1.0 + self.costmap[point[0]][point[1]])) * self.cost_heuristic(point, self.source)

    def extract_path(self):
        path_for = [self.point_meet]
        point_path_for = self.point_meet
        while True:
            point_path_for = self.explore_tree_for[point_path_for]
            path_for.append(point_path_for)
            
            if point_path_for == self.source:
                break
        
        path_back = []
        point_path_back = self.point_meet
        while True:
            point_path_back = self.explore_tree_back[point_path_back]
            path_back.append(point_path_back)
            
            if point_path_back == self.goal:
                break 

        return list(reversed(path_for)) + list(path_back)

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
                self.explore_base_for[(i, j)] = math.inf
                self.explore_tree_for[(i, j)] = None
                self.explore_base_back[(i, j)] = math.inf
                self.explore_tree_back[(i, j)] = None
        
        self.explore_base_for[self.source] = 0
        self.explore_tree_for[self.source] = self.source
        self.parent_direction[self.source] = (0, 0)
        heapq.heappush(self.open_set_for, (self.cost_total_for(self.source), self.source))

        self.explore_base_back[self.goal] = 0
        self.explore_tree_back[self.goal] = self.goal   
        self.parent_direction[self.goal] = (0, 0)
        heapq.heappush(self.open_set_back, (self.cost_total_back(self.goal), self.goal))

    def improve_path(self, flag):
        while self.open_set_for and self.open_set_back:
            cost_total_explore_point_for, explore_point_for = heapq.heappop(self.open_set_for)
            self.close_set_for.append(explore_point_for)
            
            if flag:
                if  self.explore_tree_back[explore_point_for]:
                    self.point_meet = explore_point_for
                    self.cost_total_point_meet = self.explore_base_for[self.point_meet] + self.explore_base_back[self.point_meet]
                    break
            else:
                if cost_total_explore_point_for >=  self.explore_base_for[explore_point_for]:
                    continue
                if self.explore_base_back[explore_point_for] != math.inf:
                    if self.cost_total_point_meet > self.explore_base_for[explore_point_for] + self.explore_base_back[explore_point_for]:
                        self.point_meet = explore_point_for
                        self.cost_total_point_meet = self.explore_base_for[explore_point_for] + self.explore_base_back[explore_point_for]
                        break
            
            for motion in self.motions:
                neighbor_for = (explore_point_for[0] + motion[0], explore_point_for[1] + motion[1])
                cost_neighbor_for = abs(motion[0]) + abs(motion[1]) if neighbor_for not in self.obs else math.inf

                corner_flag = self.parent_direction[explore_point_for] != motion   
                new_cost = self.explore_base_for[explore_point_for] + (self.alpha + (1.0 - self.alpha) * (1.0 + self.costmap[neighbor_for[0]][neighbor_for[1]])) * cost_neighbor_for
                if corner_flag:
                    new_cost += self.corner_loss

                if new_cost < self.explore_base_for[neighbor_for]:
                    self.explore_base_for[neighbor_for] = new_cost
                    self.explore_tree_for[neighbor_for] = explore_point_for
                    self.parent_direction[neighbor_for] = (neighbor_for[0] - explore_point_for[0],neighbor_for[1] - explore_point_for[1])

                    if neighbor_for not in self.close_set_for:
                        heapq.heappush(self.open_set_for, (self.cost_total_for(neighbor_for), neighbor_for))
                    else:
                        self.incons_set_for.append(neighbor_for)

            cost_total_explore_point_back, explore_point_back = heapq.heappop(self.open_set_back)
            self.close_set_back.append(explore_point_back)

            if flag:
                if  self.explore_tree_for[explore_point_back]:
                    self.point_meet = explore_point_back
                    self.cost_total_point_meet = self.explore_base_for[self.point_meet] + self.explore_base_back[self.point_meet]
                    break
            else:
                if cost_total_explore_point_back >= self.explore_base_back[explore_point_back]:
                    continue
                if self.explore_base_for[explore_point_back] != math.inf:
                    if self.cost_total_point_meet > self.explore_base_for[explore_point_back] + self.explore_base_back[explore_point_back]:
                        self.point_meet = explore_point_back
                        self.cost_total_point_meet = self.explore_base_for[explore_point_back] + self.explore_base_back[explore_point_back]
                        break

            for motion in self.motions:
                neighbor_back = (explore_point_back[0] + motion[0], explore_point_back[1] + motion[1])
                cost_neighbor_back = abs(motion[0]) + abs(motion[1]) if neighbor_back not in self.obs else math.inf

                corner_flag = self.parent_direction[explore_point_back] != motion        
                new_cost = self.explore_base_back[explore_point_back] + (self.alpha + (1.0 - self.alpha) * (1.0 + self.costmap[neighbor_back[0]][neighbor_back[1]])) * cost_neighbor_back
                if corner_flag:
                    new_cost += self.corner_loss

                if new_cost < self.explore_base_back[neighbor_back]:
                    self.explore_base_back[neighbor_back] = new_cost
                    self.explore_tree_back[neighbor_back] = explore_point_back
                    self.parent_direction[neighbor_back] = (neighbor_back[0] - explore_point_back[0],neighbor_back[1] - explore_point_back[1])

                    if neighbor_back not in self.close_set_back:
                        heapq.heappush(self.open_set_back, (self.cost_total_back(neighbor_back), neighbor_back))
                    else:
                        self.incons_set_back.append(neighbor_back)

        self.visited_for.append(self.close_set_for)
        self.visited_back.append(self.close_set_back)

    def searching(self):
        self.torrent()

        self.improve_path(True)
        former_path = self.extract_path()
        former_corners = self.count_corner(former_path)

        while True:
            for point in self.incons_set_for:
                    heapq.heappush(self.open_set_for, (self.cost_total_for(point), point))
            for point in self.incons_set_back:
                    heapq.heappush(self.open_set_back, (self.cost_total_back(point), point))

            self.incons_set_for = []
            self.incons_set_back = []
            self.close_set_for = []
            self.close_set_back = []

            self.improve_path(False)
            path = self.extract_path()
            corners = self.count_corner(path)

            if corners >= former_corners or len(path) >= len(former_path):
                break
            else:
                former_path = path
                former_corners = corners

        return former_path, former_corners, self.calculate_safety(former_path),  self.visited_for, self.visited_back

def main():
    BRastar = brastar()
    ploter = Plot.plot()
    BRastar.update_costmap()

    T1 = time.time()
    path, corners, safety_factor, visited_for, visited_back = BRastar.searching()

    print('search time:\t', time.time() - T1)
    print('search node:\t', len(visited_for[0]) + len(visited_back[0]))
    print('corner count:\t', corners)
    print('safty_factor:\t', safety_factor)
    print('path_length:\t', len(path))

    ploter.animation("BRAstar", path, False, "BRAstar", visited_for, visited_back)
    
    plt.show()

if __name__ == '__main__':
    main()