"""
A* implementation with crappy visualization

Fischer


"""
import numpy as np
from enum import Enum
from queue import PriorityQueue
import math



class Action(Enum):
    LEFT  = (0, -1, 1)
    RIGHT = (0, 1, 1)
    UP    = (-1, 0, 1)
    DOWN  = (1, 0, 1)

    def __str__(self):
        if self == self.LEFT:
            return '<'
        elif self == self.RIGHT:
            return '>'
        elif self == self.UP:
            return '^'
        elif self == self.DOWN:
            return 'v'
    # Make a new property that returns the cost
    @property
    def cost(self):
        return self.value[2]

    # Assign a property that returns the action itself
    @property
    def delta(self):
        return (self.value[0], self.value[1])

def valid_actions(grid, current_node):
    """
    Remove actions that arent valid based on node location

    """

    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN]
    n, m = grid.shape[0] - 1, grid.shape[1] -  1
    x, y = current_node

    # Get rid of nodes that are off-the-grid or are an obstacle.
    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[x+1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[x, y-1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid.remove(Action.RIGHT)
    return valid

def heuristic(position, goal):
    # Let's use a euclidian heuristic
    h = 0.0
    h = math.sqrt((goal[0] - position[0])**2 + (goal[1] - position[1])**2)
    return h


def a_star(grid, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path bruh')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the valid valid_actions
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                h = heuristic(current_node,goal)
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        n = goal
        path_cost = branch[n][0]
        path = []
        while branch[n][1] != start:
            print(branch[n])
            path.append(branch[n][2])
            n = branch[n][1]
        path.append(branch[n][2]) # Add start node after while loop runs
    return path[::-1], path_cost

# Define a function to visualize the path
def visualize_path(grid, path, start):
    sgrid = np.zeros(np.shape(grid), dtype=np.str)
    sgrid[:] = ' '
    sgrid[grid[:] == 1] = 'O'

    pos = start

    for a in path:
        da = a.value
        sgrid[pos[0], pos[1]] = str(a)
        pos = (pos[0] + da[0], pos[1] + da[1])
    sgrid[pos[0], pos[1]] = 'G'
    sgrid[start[0], start[1]] = 'S'
    return sgrid


def main():
    # Define start and goal location
    start = (0, 0)
    goal  = (4, 4)

    # Define grid-based state space of obstacles
    grid = np.array([
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
    ])

    path, path_cost = a_star(grid,start,goal)
    print(path)
    sgrid = visualize_path(grid, path, start)
    print(sgrid)
    print('Cost is: ',path_cost)

if __name__ == "__main__":
    main()
