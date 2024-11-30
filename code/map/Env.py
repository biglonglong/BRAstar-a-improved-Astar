from map import Map

class env:
    def __init__(self):
        self.map = Map.map5
        self.range_x, self.range_y = self.map.shape
        self.obs = self.position_obs()
        
        self.source = Map.source5
        self.goal = Map.goal5
        
    def position_obs(self):
        x = self.range_x
        y = self.range_y
        obs = set()

        for i in range(x):
            for j in range(y):
                if self.map[i][j]:
                    obs.add((i, j))

        return obs