import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from map import Env

class plot:
    def __init__(self):
        self.env = Env.env()
        self.obs = self.env.obs 

        self.source = self.env.source
        self.goal = self.env.goal

        self.ims = [[]]

    def plot_env(self, title):
        base_obs_x = [obs[0] for obs in self.env.obs]
        base_obs_y = [obs[1] for obs in self.env.obs]

        plt.title(title)
        plt.axis('equal')

        plt.plot(self.source[0], self.source[1], color='blue', marker='s')
        plt.plot(self.goal[0], self.goal[1], color='green', marker='s')
        plt.plot(base_obs_x, base_obs_y, 'ks')

        plt.pause(1.0)

    def plot_visited(self, color_visited, *args):
        counter = 0
        length = 40
        im_explore_points = []
        
        if self.source in args[0]:
            args[0].remove(self.source)
        if self.goal in args[0]:
            args[0].remove(self.goal)

        if len(args) == 1:
            for point in args[0]:
                counter += 1
                im_explore_point = plt.plot(point[0], point[1], color=color_visited, marker='s')
                im_explore_points = im_explore_points + im_explore_point

                if counter % length == 0 or counter == len(args[0]):
                    self.ims.append(self.ims[-1] + im_explore_points)
                    plt.pause(0.01)
                    
        else:
            if self.source in args[1]:
                args[1].remove(self.source)
            if self.goal in args[1]:
                args[1].remove(self.goal)            

            len_visited_for, len_visited_back = len(args[0]), len(args[1])
            for i in range(max(len_visited_for, len_visited_back)):
                if i < len_visited_for:
                    counter += 1
                    im_explore_point_for = plt.plot(args[0][i][0], args[0][i][1], color=color_visited, marker='s')
                    im_explore_points = im_explore_points + im_explore_point_for
                if i < len_visited_back:
                    counter += 1
                    im_explore_point_back = plt.plot(args[1][i][0], args[1][i][1], color=color_visited, marker='s')
                    im_explore_points = im_explore_points + im_explore_point_back

                if counter % length == 0 or counter == len(args[0]+args[1]):
                    self.ims.append(self.ims[-1] + im_explore_points)
                    plt.pause(0.001)

    def plot_path(self, path):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        
        im_route = plt.plot(path_x, path_y, color='red', linewidth='2')
        self.ims.append(self.ims[-1] + im_route)
        plt.pause(1.0)

    def animation(self, title, path, button = True, file='test', *args):
        plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
        
        self.plot_env(title)

        if len(args) == 2:
            if isinstance(args[0][0], list):
                self.plot_visited('darkgrey', args[0][0], args[1][0])
                for k in range(1, len(args[0])):
                    self.plot_visited('lightgray', args[0][k], args[1][k])
            else:
                self.plot_visited('darkgrey', args[0], args[1]) 
            
        elif len(args) == 1:
            if isinstance(args[0][0], list):
                self.plot_visited('darkgrey', args[0][0])
                for k in range(1, len(args[0])):
                    self.plot_visited('lightgray', args[0][k])
            else:
                self.plot_visited('darkgrey', args[0])

        else:
            pass  

        self.plot_path(path)
            
        if button:
            ani = animation.ArtistAnimation(plt.gcf(), self.ims, interval=100,
                                                repeat_delay=1000, blit=True)
            ani.save(os.path.dirname(os.path.abspath(__file__)) + rf'\gif\{file}.gif',
                        writer='pillow')