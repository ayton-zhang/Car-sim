import math
import random

import sys
import matplotlib.pyplot as plt
import cv2
import pathlib

SHOW_SAMPLING_PROCESS = True

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
            print("Iteration:", i, ", node list nums:", len(self.node_list))
            # TODO: 1. generate random node within the x/y range
            sampled_node = self.generate_random_node()

            # TODO: 2. find the nearest node to random node in the tree
            nearest_idx = self.get_nearest_node_index(self.node_list, sampled_node)
            nearest_node = self.node_list[nearest_idx]

            # TODO: 3. Steer the nearest_node to random node with extend_dist
            new_node = self.steer(nearest_node, sampled_node, self.extend_dist)
            new_node.cost = nearest_node.cost +  math.hypot(new_node.x - nearest_node.x, \
                                                            new_node.y - nearest_node.y)

            if self.check_collision_with_obs_map(new_node, self.obstacle_map):
                # TODO: 4. find nodes near the new_node, and choose the parent which maintains 
                # a minimum-cost path from the start node
                near_idxs = self.find_near_nodes(new_node)
                updated_node = self.choose_parent(new_node, near_idxs)

                # TODO: 5. rewire the tree and draw current node and the edge between it and its parent node
                if updated_node:
                    self.rewire(updated_node, near_idxs)
                    self.node_list.append(updated_node)
                    self.draw_node_and_edge(updated_node)
                else:
                    self.node_list.append(new_node)
                    self.draw_node_and_edge(updated_node)

            # TODO: 6. check if the tree has expanded to the vicinity of the goal point, if so, select a suitable
            # node directly to connect to the goal, if the edge does not collide with obstacles, the search can 
            # be finished, return the entire path which is extracted by backtracking.
            if ((not self.search_until_max_iter) and new_node):
                # determine if the search can be finished
                last_index = self.find_best_node_around_goal()
                if last_index is not None:
                    return self.backtracking(last_index)

        print("reach max iteration")
        # TODO: 7. reach max iteration, backtrack to extract the path between the last node and the start node
        last_index = self.find_best_node_around_goal()
        if last_index is not None:
            return self.backtracking(last_index)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        # TODO: Steer the 'from_node' to 'to_node' with 'extend_length' and record the edge between 
        # these two nodes. you should construct a 'Node' class and return it.
        new_node = self.Node(from_node.x, from_node.y)
        dist, theta = self.cal_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if dist < extend_length:
            extend_length = dist

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        dist, _ = self.cal_distance_and_angle(new_node, to_node)
        if dist <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node
        
    def choose_parent(self, new_node, near_idxs):
        """
        Finds the lowest cost node to new_node contained in the list
        near_idxs and set this node as the parent of new_node.
        """
        if not near_idxs:
            return None

        # find smallest cost node in near_idxs
        costs = []
        for idx in near_idxs:
            near_node = self.node_list[idx]
            tmp_node = self.steer(near_node, new_node)

            # need to check whether the edge between near node and new node is valid
            if tmp_node and self.check_collision_with_obs_map(tmp_node, self.obstacle_map):
                costs.append(self.calculate_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # current edge is collision
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("min_cost is inf, no valid edge founded")
            return None

        min_cost_idx = near_idxs[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_cost_idx], new_node)
        new_node.cost = min_cost

        return new_node

    def find_near_nodes(self, new_node):
        # TODO:  Defines a circle centered on new_node and returns all nodes of 
        # current tree that are inside this circle, return the 
        radius = self.near_nodes_dist_threshold
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_idxs = [dist_list.index(dist) for dist in dist_list if dist <= radius ** 2]

        return near_idxs

    def rewire(self, new_node, near_idxs):
        """
            For each node in near_idxs, this function will check if it is cheaper to
            arrive start point from new_node. If go this way, the cost will be smaller,
            we will re-assign the parent of nodes in near_idxs to new_node. In addition, 
            it is necessary to propagate the cost of the new node to others until find
            the leaf node.
        """
        for idx in near_idxs:
            near_node = self.node_list[idx]
            rewired_node = self.steer(new_node, near_node)
            if not rewired_node:
                continue
            rewired_node.cost = self.calculate_cost(new_node, near_node)

            if self.check_collision_with_obs_map(rewired_node, self.obstacle_map) \
                                        and near_node.cost > rewired_node.cost:
                near_node.x = rewired_node.x
                near_node.y = rewired_node.y
                near_node.cost = rewired_node.cost
                near_node.path_x = rewired_node.path_x
                near_node.path_y = rewired_node.path_y
                near_node.parent = rewired_node.parent
                self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                # update node cost
                node.cost = self.calculate_cost(parent_node, node)
                # recursively propagate until find the leaf node
                self.propagate_cost_to_leaves(node)

    def calculate_cost(self, from_node, to_node):
        dist, _ = self.cal_distance_and_angle(from_node, to_node)
        return from_node.cost + dist

    def cal_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def generate_random_node(self):
        # TODO: generate random node within the x/y range, return the random node
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_node = self.Node(
                random.uniform(self.x_min_rand, self.x_max_rand),
                random.uniform(self.y_min_rand, self.y_max_rand))
        else:  # goal point sampling
            rand_node = self.Node(self.end.x, self.end.y)
        return rand_node

    def find_best_node_around_goal(self):
        """
        Find nodes around the goal, and steer these nodes to goal, 
        if it is safety, stop searching and choose the node with min cost.
        """
        dist_to_goal_list = [
            self.cal_dist_to_goal(node.x, node.y) for node in self.node_list
        ]
        goal_indexs = [
            dist_to_goal_list.index(dist) for dist in dist_to_goal_list
            if dist <= self.extend_dist
        ]

        safe_goal_indexs = []
        for goal_index in goal_indexs:
            tmp_node = self.steer(self.node_list[goal_index], self.goal_node)
            if self.check_collision_with_obs_map(tmp_node, self.obstacle_map):
                safe_goal_indexs.append(goal_index)

        # unable to find a suitable node near the goal to stop searching
        if not safe_goal_indexs:
            return None

        # find a suitable node near the goal to terminate the searching 
        # and choose the node with the lowest cost
        min_cost = min([self.node_list[idx].cost for idx in safe_goal_indexs])
        for idx in safe_goal_indexs:
            if self.node_list[idx].cost == min_cost:
                return idx

        return None

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

    def get_nearest_node_index(self, node_list, rand_node):
        # TODO: find the nearest node to rand_node in the tree, return the min_dist node index
        dist_list = [(node.x - rand_node.x)**2 + (node.y - rand_node.y)**2
                    for node in node_list]
        min_dist_idx = dist_list.index(min(dist_list))

        return min_dist_idx

    def check_collision_with_obs_map(self, node, obstacle_map):
        if node is None:
            return False

        for x, y in zip(node.path_x, node.path_y):
            if (obstacle_map[round(x)][round(y)] == True):
                return False  # collision

        return True  # safe

    def cal_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return dist, theta


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
    image = cv2.imread(str(pathlib.Path.cwd()) + "/maps/" + "map3.png")
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
