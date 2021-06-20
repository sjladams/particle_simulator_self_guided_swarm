import numpy as np
from configuration import *

class Grid:
    def __init__(self, obstacles=None):
        self.obstacles = obstacles

        # self.ref_lines_domain = np.array([[[0, 0], [0, domain[1]]],
        #                              [[0, domain[1]], [domain[0], domain[1]]],
        #                              [[domain[0], domain[1]], [domain[0], 0]],
        #                              [[domain[0], 0], [0, 0]]])
        # self.ref_lines_domain = np.array([
        #                                 [[domain[0][0], domain[1][0]], [domain[0][0], domain[1][1]]],
        #                                 [[domain[0][0], domain[1][1]], [domain[0][1], domain[1][1]]],
        #                                 [[domain[0][1], domain[1][1]], [domain[0][1], domain[1][0]]],
        #                                 [[domain[0][1], domain[1][0]], [domain[0][0], domain[1][0]]]
        #                                 ])
        self.ref_lines_domain = np.array([
            [domain[0], domain[1]],
            [domain[1], domain[2]],
            [domain[2], domain[3]],
            [domain[3], domain[0]]
        ])
        self.ref_lines = self.ref_lines_domain
        if self.obstacles:
            for item in self.obstacles:
                ref_lines_obstacle = np.array([
                    [self.obstacles[item][0], self.obstacles[item][3]],
                    [self.obstacles[item][3], self.obstacles[item][2]],
                    [self.obstacles[item][2], self.obstacles[item][1]],
                    [self.obstacles[item][1], self.obstacles[item][0]]
                ])
                self.ref_lines = np.concatenate((self.ref_lines, ref_lines_obstacle), axis=0)

        # if self.obstacle:
        #     self.ref_lines_obstacle = np.array([[obstacle[0], obstacle[3]],
        #                                    [obstacle[3], obstacle[2]],
        #                                    [obstacle[2], obstacle[1]],
        #                                    [obstacle[1], obstacle[0]]])
        #     self.ref_lines = np.concatenate((self.ref_lines_domain, self.ref_lines_obstacle), axis=0)
        # else:
        #     self.ref_lines = self.ref_lines_domain

    @staticmethod
    def normalize(item):
        return_value = item / np.linalg.norm(item)
        if np.isnan(return_value).any():
            print('return value in to normalize value')
        else:
            return return_value

    @staticmethod
    def perpendicular(a):
        b = np.empty_like(a)
        b[0] = a[1]
        b[1] = -a[0]
        return b

    @staticmethod
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def check_direction_vectors(vector_1, vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        return dot_product > 0

    def check_in_domain(self,point):
        if (point[0] > domain[0][0] and point[0] < domain[2][0] and \
                point[1] > domain[0][1] and point[1] < domain[1][1]):
            if self.obstacles:
                for item in self.obstacles:
                    if (point[0] > self.obstacles[item][0][0] and point[0] < self.obstacles[item][3][0] and \
                            point[1] > self.obstacles[item][0][1] and point[1] < self.obstacles[item][1][1]):
                        # point within obstacle area:
                        return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def line_intersection(self,ant_locations):
        pot_inter = {}
        move_vec = np.round(np.array([ant_locations[1][0] - ant_locations[0][0],
                                      ant_locations[1][1] - ant_locations[0][1]]), 3)

        for count, ref_line in enumerate(self.ref_lines):
            xdiff = np.array([ref_line[0][0] - ref_line[1][0], ant_locations[0][0] - ant_locations[1][0]])
            ydiff = np.array([ref_line[0][1] - ref_line[1][1], ant_locations[0][1] - ant_locations[1][1]])

            div = self.det(xdiff, ydiff)
            if div != 0:
                d = (self.det(*ref_line), self.det(*ant_locations))
                inter = np.array([self.det(d, xdiff) / div, self.det(d, ydiff) / div])

                inter_vec = np.round(np.array([inter[0] - ant_locations[0][0],
                                               inter[1] - ant_locations[0][1]]), 3)

                if self.check_direction_vectors(move_vec, inter_vec):
                    pot_inter[count] = {'inter': inter,
                                        'inter_dist': np.linalg.norm(inter_vec)}

        if pot_inter:
            key_min = min(pot_inter.keys(), key=(lambda k: pot_inter[k]['inter_dist']))
            return key_min, pot_inter[key_min]['inter']
        else:
            return None, None

    def obstacle_avoidance(self,start_point, move):
        no_obs_new_point = start_point + move

        if not self.check_in_domain(start_point + move):
            if np.linalg.norm(start_point + move) < step_threshold:
                print('hier')

            index_line, inter = self.line_intersection(np.array([start_point, start_point + move]))
            if isinstance(index_line, int):
                ref_line = self.ref_lines[index_line]
                ref_vec = np.array([ref_line[1][0] - ref_line[0][0], ref_line[1][1] - ref_line[0][1]])

                per_vec = self.perpendicular(ref_vec)
                if not per_vec[0]:
                    new_point = np.array([no_obs_new_point[0], 2 * inter[1] - move[1] - start_point[1]])
                else:
                    new_point = np.array([2 * inter[0] - move[0] - start_point[0], no_obs_new_point[1]])

                # Check if mirrored point is within region (we do not calculate double bouncing)
                if self.check_in_domain(new_point):
                    return new_point, True
                else:
                    # if mirrored point is not within region of influence, try to move the opposite direction
                    new_point_alt = start_point - move
                    if self.check_in_domain(new_point_alt):
                        # print('mirrored location is not feasible, move opposite direction')
                        return new_point_alt, True

                    # if this also doesn't result in an feasible location, stay were you are
                    else:
                        print('robot is stuck, stays at same location')
                        return start_point, True
            else:
                return start_point + move, False
        else:
            return start_point + move, False

def mapper(fnc):
    def inner(x, y, beacons,w_type):
        value = 0
        for beac_tag in beacons:
            value += fnc(x, y, beacons[beac_tag],w_type)
        return value
    return inner

