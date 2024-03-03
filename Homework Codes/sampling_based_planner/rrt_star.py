import math
import random

import sys
import matplotlib.pyplot as plt
import cv2
import pathlib
import copy

SHOW_SAMPLING_PROCESS = True
random.seed(1)
class RRTStar:

    class Node:
        def __init__(self, x, y):
            # the position of node
            self.x = x
            self.y = y
            # the path between current node and its parent
            self.path_x = []
            self.path_y = []
            # the cost of node
            self.cost = 0.0
            # the parent of node
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 x_rand_area,
                 y_rand_area,
                 obstacle_map=None,
                 extend_dist=10.0,
                 path_resolution=1.0,
                 goal_sample_rate=10,
                 max_iter=1000,
                 near_nodes_dist_threshold=30.0,
                 search_until_max_iter=False):

        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.x_min_rand = x_rand_area[0]
        self.x_max_rand = x_rand_area[1]
        self.y_min_rand = y_rand_area[0]
        self.y_max_rand = y_rand_area[1]
        self.obstacle_map = obstacle_map
        self.extend_dist = extend_dist
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.near_nodes_dist_threshold = near_nodes_dist_threshold
        self.search_until_max_iter = search_until_max_iter

        self.goal_node = self.Node(goal[0], goal[1])

        self.node_list = []

        
    def planning(self):
        self.draw_start_and_goal()
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # TODO: 1. generate random node within the x/y range
            random_node = self.generate_random_node()

            
            # TODO: 2. find the nearest node to random node in the tree
            nearest_node = min(self.node_list, key=lambda node: math.hypot(node.x - random_node.x, node.y - random_node.y))


            # TODO: 3. Steer the nearest_node to random node with extend_dist
            dx = random_node.x - nearest_node.x
            dy = random_node.y - nearest_node.y
            d = math.hypot(dx, dy)
            if d > self.path_resolution:
                if d < self.extend_dist:
                    self.extend_dist = d
                dx_extend = (dx / d) * self.extend_dist
                dy_extend = (dy / d) * self.extend_dist
                new_node = self.Node(nearest_node.x + dx_extend, nearest_node.y + dy_extend)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.extend_dist
                mid_node = self.generate_path_point(nearest_node, new_node, self.extend_dist, dx_extend, dy_extend)
                new_node = mid_node
            if d <= self.path_resolution:
                new_node = random_node
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + d
                new_node.path_x = [nearest_node.x, random_node.x]
                new_node.path_y = [nearest_node.y, random_node.y]
            # for i in range(steps):
            #     new_node.path_x.append(nearest_node.x + self.path_resolution * i)
            #     new_node.path_y.append(nearest_node.y + self.path_resolution * i)
            # else:
            #     new_node = random_node
            #     new_node.parent = nearest_node
            #     new_node.cost = nearest_node.cost + d
            #     self.generate_path_point(new_node, nearest_node, d)
                # for i in range(steps):
                #     new_node.path_x.append(nearest_node.x + self.path_resolution * i)
                #     new_node.path_y.append(nearest_node.y + self.path_resolution * i)

            # check if the new node collides with obstacles
            if self.check_collision_with_obs_map(new_node, self.obstacle_map):
                # TODO: 4. find nodes near the new_node, and choose the parent which maintains 
                # a minimum-cost path from the start node
                near_nodes=[]
                for node in self.node_list:
                    if math.hypot(node.x - new_node.x, node.y - new_node.y) <= self.near_nodes_dist_threshold:
                        near_nodes.append(node)
                min_cost = new_node.cost
                for node in near_nodes:
                    newnew_node = self.generate_path_point(node, new_node, math.hypot(node.x - new_node.x, node.y - new_node.y), (new_node.x - node.x), (new_node.y - node.y))
                    if self.check_collision_with_obs_map(newnew_node, self.obstacle_map):
                        if node.cost + math.hypot(node.x - newnew_node.x, node.y - newnew_node.y) < min_cost:
                            newnew_node.cost = node.cost + math.hypot(node.x - newnew_node.x, node.y - newnew_node.y)
                            min_cost = newnew_node.cost
                            newnew_node.parent = node
                new_node = newnew_node
                self.node_list.append(new_node)
                self.draw_node_and_edge(new_node)

                # TODO: 5. rewire the tree and draw current node and the edge between it and its parent node
                for node in near_nodes:
                    rewire_node = self.generate_path_point(new_node, node, math.hypot(node.x - new_node.x, node.y - new_node.y), (node.x - new_node.x), (node.y - new_node.y))
                    if self.check_collision_with_obs_map(rewire_node, self.obstacle_map):
                        if new_node.cost + math.hypot(rewire_node.x - new_node.x, rewire_node.y - new_node.y) < rewire_node.cost:
                            rewire_node.parent = new_node
                            rewire_node.cost = new_node.cost + math.hypot(rewire_node.x - new_node.x, rewire_node.y - new_node.y)
                            node = rewire_node
                            self.node_list.append(node)
                            self.draw_node_and_edge(node)
            

            # TODO: 6. check if the tree has expanded to the vicinity of the goal point, if so, select a suitable
            # node directly to connect to the goal, if the edge does not collide with obstacles, the search can 
            # be finished, return the entire path which is extracted by backtracking.
            if ((not self.search_until_max_iter) and new_node):
                if math.hypot(new_node.x - self.end.x, new_node.y - self.end.y) <= self.extend_dist:
                    mid_end_node = self.generate_path_point(new_node, self.end, math.hypot(new_node.x - self.end.x, new_node.y - self.end.y), (self.end.x - new_node.x), (self.end.y - new_node.y))
                    if self.check_collision_with_obs_map(mid_end_node, self.obstacle_map):
                        mid_end_node.parent = new_node
                        mid_end_node.cost = new_node.cost + math.hypot(new_node.x - self.end.x, new_node.y - self.end.y)
                        self.end = mid_end_node
                        self.node_list.append(self.end)
                        return self.backtracking(len(self.node_list) - 1)


        print("reach max iteration")
        # TODO: 7. reach max iteration, backtracking to extract the path between the last node and the start node
        return self.backtracking(len(self.node_list) - 1)


    def generate_path_point(self, from_node, to_node, distance, dx, dy):
        to_node = copy.deepcopy(to_node)
        to_node.path_x = [from_node.x]
        to_node.path_y = [from_node.y]
        new_node = self.Node(from_node.x, from_node.y)
        n_expand = math.floor(distance / self.path_resolution)

        for _ in range(n_expand):
            new_node.x = new_node.x + self.path_resolution * dx / distance
            new_node.y = new_node.y + self.path_resolution * dy / distance
            to_node.path_x.append(new_node.x)
            to_node.path_y.append(new_node.y)
        
        return to_node
        

    def generate_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_node = self.Node(
                random.uniform(self.x_min_rand, self.x_max_rand),
                random.uniform(self.y_min_rand, self.y_max_rand))
        else:  # goal point sampling
            rand_node = self.Node(self.end.x, self.end.y)
        return rand_node


    def backtracking(self, goal_idx):
        """
            backtracking from last index to retrieve the entire path.
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_idx]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def check_collision_with_obs_map(self, node, obstacle_map):
        if node is None:
            return False

        for x, y in zip(node.path_x, node.path_y):
            if (obstacle_map[round(x)][round(y)] == True):
                return False  # collision

        return True  # safe

    def draw_start_and_goal(self):
        plt.plot(self.start.x, self.start.y, marker='*', color='darkviolet', markersize=8, zorder=100000)
        plt.plot(self.end.x, self.end.y, marker='*', color='r', markersize=8, zorder=100000)

    def draw_node_and_edge(self, node):
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(node.x, node.y, '.', color='cornflowerblue')
        if node.parent:
            plt.plot(node.path_x, node.path_y, '-', color='lightpink')
        if (SHOW_SAMPLING_PROCESS):
            plt.pause(0.00000001)

def preprocess_image(image, threshold):
    # convert to gray image
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # binarization
    _, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def extract_obstacle_list_from_img(binary_img):
    obstacle_x_list = []
    obstacle_y_list = []
    # get the size of binary image
    rows, cols = binary_img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 0:
                # convert image frame to world frame
                obstacle_x_list.append(j)
                obstacle_y_list.append(rows - i - 1)

    return obstacle_x_list, obstacle_y_list

def extract_obstacle_map_list_from_img(binary_img):
    # get the size of binary image
    img_rows, img_cols = binary_img.shape[:2]

    # convert image frame to world frame
    map_x_size = img_cols
    map_y_size = img_rows

    # contruct obstacle map
    obstacle_map = [[False for y in range(map_y_size)] for x in range(map_x_size)]

    for x in range(map_x_size):
        for y in range(map_y_size):
            if binary_img[map_y_size - y - 1, x] == 0:
                obstacle_map[x][y] = True

    return obstacle_map

def main():
    # read image map
    image = cv2.imread(r'D:\Car-sim\Homework Codes\sampling_based_planner\maps\map3.png')
    # transform to binary image
    binary_img = preprocess_image(image, 127)
    # extract obstacle map from image
    obstacle_map = extract_obstacle_map_list_from_img(binary_img)

    x_size = len(obstacle_map)
    y_size = len(obstacle_map[0])

    obs_x_list, obs_y_list = [], []
    for x in range(len(obstacle_map)):
        for y in range(len(obstacle_map[0])):
            if  obstacle_map[x][y] == True:
                obs_x_list.append(x)
                obs_y_list.append(y)

    plt.figure(figsize=(12, 12))
    plt.axis("equal")
    plt.plot(obs_x_list, obs_y_list, 'sk', markersize=2)

    rrt_star = RRTStar(
        start=[20, 20],
        goal=[250, 210],
        x_rand_area=[0, len(obstacle_map)],
        y_rand_area=[0, len(obstacle_map[0])],
        obstacle_map=obstacle_map,
        extend_dist=8.0,
        max_iter=3000,
        search_until_max_iter=False)

    path = rrt_star.planning()

    if path is None:
        print("Cannot find path!")
    else:
        print("Find path!")

    # draw final path
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-', linewidth=2.0, color='palegreen')
    plt.show()

if __name__ == '__main__':
    main()
