import math
import numpy as np



def select_next_goal(state, current_goal, path,radius = 0.05):
    for pose in range(len(path)-1):
        dist = math.sqrt(math.pow(state[0] - path[pose,0], 2) + math.pow(state[1] - path[pose,1], 2))
        if dist<radius:
            return path[pose+1]
    return current_goal

class GlobalPlanner:

    def __init__(self):
        pass

    def get_nearest_goal(self, path, position):
        pass

    def get_neighbors(self, position, grid):
        # add neighbors
        neighbors =np.array([
            [position[0]-1, position[1]],
            [position[0]-1, position[1]-1],
            [position[0], position[1]-1],

            [position[0]+1, position[1]],
            [position[0]+1, position[1]+1],
            [position[0], position[1]+1],

            [position[0]-1, position[1]+1],
            [position[0]+1, position[1]-1],
        ])
        # remove out of map
        p=0
        for n in range(len(neighbors)):
            if (neighbors[-1-p] <= 0).any() or (neighbors[-1-p] >= grid.shape).any():
                neighbors = np.delete(neighbors,-1-p,0) #neighbors.pop([-1-n])
                continue
            p+=1
        return neighbors

    def line_Of_sight(self, point1, point2, grid):
        # n = 4 ?
        # n = 4
        # dxy = (np.sqrt((point1[0] - point2[0]) **
        #                2 + (point1[1] - point2[1]) ** 2)) * n
        # i = np.rint(np.linspace(point1[0], point2[0], dxy)).astype(int)
        # j = np.rint(np.linspace(point1[1], point2[1], dxy)).astype(int)
        # dxy = max(abs(point1[0]-point2[0])+1,abs(point1[1]-point2[1])+1)
        dxy = self.heuristic_manhetten(point1,point2)
        i = np.linspace(point1[0], point2[0],dxy).astype(int)
        j = np.linspace(point1[1], point2[1],dxy).astype(int)
        has_collision = np.any(grid[i, j])
        return has_collision

    def heuristic_evclid(self, source, target):
        return math.sqrt(math.pow(source[0] - target[0], 2) + math.pow(source[1] - target[1], 2))

    def heuristic_chebishev(self, source, target):
        a = abs(source[0] - target[0])
        b = abs(source[1] - target[1])
        if b > a:
            a = b
        return a

    def heuristic_manhetten(self, source, target):
        return (abs(source[0] - target[0]) + abs(source[1] - target[1]))


class GlobalPlannerThetaStar(GlobalPlanner):
    def __init__(self):
        pass

    def calc_path(self, start, goal, grid):
        # grid = grid.astype(bool)
        open_set = [start]
        cost_so_far = {hash(start.tostring()): 0}
        came_from = {hash(start.tostring()): start}
        while len(open_set) > 0:
            current = open_set.pop()  # check
            parent = came_from[hash(current.tostring())]
            if (current == goal).all():
                break
            neighbors = self.get_neighbors(current, grid)
            current_cost = cost_so_far[hash(current.tostring())]
            parent_cost = cost_so_far[hash(parent.tostring())]
            for neighbor in neighbors:
                if hash(neighbor.tostring()) not in cost_so_far:
                    cost_so_far[hash(neighbor.tostring())] = 999999999999
                neighbor_cost = cost_so_far[hash(neighbor.tostring())]
                has_collision = self.line_Of_sight(parent, neighbor, grid)
                if has_collision:
                    new_cost = current_cost + \
                        self.heuristic_evclid(current, neighbor)
                    if new_cost < neighbor_cost:
                        cost_so_far[hash(neighbor.tostring())] = new_cost
                        came_from[hash(neighbor.tostring())] = current
                        open_set.append(neighbor)
                else:
                    new_cost = parent_cost + \
                        self.heuristic_evclid(parent, neighbor)
                    if new_cost < neighbor_cost:
                        cost_so_far[hash(neighbor.tostring())] = new_cost
                        came_from[hash(neighbor.tostring())] = parent
                        open_set.append(neighbor)
        current = goal
        path = [current]
        try:
            came_from[hash(current.tostring())]
        except:
            path.append(start)
            return np.array(path)
        while (current != start).all():
            current = came_from[hash(current.tostring())]
            path.append(current)
        return np.array(path)


    # def calc_path(self, start, goal, grid):
    #     # grid = grid.astype(bool)
    #     open_set = [start]
    #     cost_so_far = {hash(start.tostring()): 0}
    #     came_from = {start: start}
    #     while len(open_set) > 0:
    #         current = open_set.pop()  # check
    #         parent = came_from[current]
    #         if current == goal:
    #             break
    #         neighbors = self.get_neighbors(current, grid)
    #         current_cost = cost_so_far[current]
    #         parent_cost = cost_so_far[parent]
    #         for neighbor in neighbors:
    #             neighbor_cost = cost_so_far[neighbor]
    #             has_collision = self.line_Of_sight(parent, neighbor, grid)
    #             if has_collision:
    #                 new_cost = current_cost + \
    #                     self.heuristic_evclid(current, neighbor)
    #                 if new_cost < neighbor_cost:
    #                     cost_so_far[neighbor] = new_cost
    #                     come_from[neighbor] = current
    #                     open_set.append(neighbor)
    #             else:
    #                 new_cost = parent_cost + \
    #                     self.heuristic_evclid(parent, neighbor)
    #                 if new_cost < neighbor_cost:
    #                     cost_so_far[neighbor] = new_cost
    #                     come_from[neighbor] = parent
    #                     open_set.append(neighbor)
    #     current = goal
    #     path = [current]
    #     if came_from[current] is None:
    #         return path
    #     while current != start:
    #         current = came_from[current]
    #         path.append(current)
    #     return path


# planner = GlobalPlannerThetaStar()
# exit()
