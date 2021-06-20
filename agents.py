from domain import *
import random
from configuration import *
import numpy as np
import struct
import array
import pickle

STD_SPEED_TRANS = 28

def angle_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

class Agent:
    def __init__(self, location, agent_tag,
                 supervisor, timestep):
        self.pt = np.array([location, location])
        self.w = np.array([0., 0.])
        self.v = np.array([[0., 0.],[0., 0.]])

        self.move = np.array([[0., 0.], [0.,0.]])
        self.mode = [0, 0]  # [node_t, node_t1] nt1 has been chosen under mode_t
        self.state = [1, 1] # state 0 is beacon, state 1 is foraging
        self.trips = 0

        self.neigh = []
        self.neigh_foragers = []
        self.neigh_beacons = []
        self.neigh_toll = []
        self.neigh_foragers_toll = []
        self.neigh_beacons_toll = []

        self.agent_tag = agent_tag
        self.range = clip_range

        self.obs_avoid_mode = False
        self.exploring_mode = True

        # CONTROLLER PART
        self.robot_name = 'ELISA3-' + str(self.agent_tag)
        self.node = supervisor.getFromDef(self.robot_name)
        self.emitter = self.node.getField('emitter_channel')
        self.receiver = self.node.getField('receiver_channel')

        self.trans_field = self.node.getField('translation')
        self.rot_field = self.node.getField('rotation')

        self.emitter.setSFInt32(self.agent_tag)
        self.receiver.setSFInt32(self.agent_tag)

        self.super_receiver = supervisor.getDevice('receiver-' + str(self.agent_tag))
        self.super_receiver.enable(timestep)
        self.super_receiver.setChannel(self.agent_tag)

        self.move_webots = np.array([0,0]) *dt
        self.get_webots_location()

        self.txData = [0 for _ in range(0, 10)]
        self.orien_motionState = 0
        self.orien_controlSteps = 0
        self.trans_motionState = 1
        self.trans_controlSteps = 0
        self.waiting_time = 0

        self.moving_beacon = False

    def get_webots_location(self):
        trans = np.array(self.trans_field.getSFVec3f())
        self.pt_webots = np.array([trans[0], trans[2]])
        rot = self.rot_field.getSFRotation()
        self.yaw_webots = rot[3] % (2*np.pi)

    def set_webots_msg(self, pol=np.array([0,0])):
        phi = pol[1] - self.yaw_webots
        if phi < 0:
            phi += (2*np.pi)
        phi %= (2*np.pi)

        if phi <= 0.5*np.pi:
            # turn
            self.orien_motionState = 0
            self.orien_controlSteps = phi * (180 / np.pi)
            self.trans_motionState = 1
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
        elif phi <= np.pi:
            self.orien_motionState = 1
            self.orien_controlSteps = (np.pi - phi) * (180 / np.pi)
            self.trans_motionState = 0
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
        elif phi <= 1.5*np.pi:
            # turn
            self.orien_motionState = 0
            self.orien_controlSteps = (phi - np.pi) * (180 / np.pi)
            self.trans_motionState = 0
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
        else:
            # turn
            self.orien_motionState = 1
            self.orien_controlSteps = (2*np.pi - phi) * (180 / np.pi)
            self.trans_motionState = 1
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS

    def send_webots_msg(self, emitter):
        self.txData = [0 for _ in range(0, 10)]
        # Send States and Modes
        if self.moving_beacon:
            self.txData[1] = int(5)
            self.moving_beacon = False
        else:
            self.txData[1] = int(self.state[1])

        self.txData[2] = int(self.mode[1])

        if self.state[1] == 0 and not beacons_move:
            self.waiting_time = 0
        else:
            self.waiting_time = int(self.trans_controlSteps * 1.5 / (WEBOTS_BASIC_TIME * 1e-6))

            self.txData[3] = int(self.orien_controlSteps)
            self.txData[4] = self.orien_motionState
            self.txData[5] = self.trans_motionState

            steps2com = str(int((self.trans_controlSteps) / (WEBOTS_BASIC_TIME * 1e-6)))
            list2com = list(reversed([steps2com[i:(i+2)] for i in range(0,len(steps2com),2)]))
            self.txData[6:] = [0, 0, 0, 0]
            for count, elem in enumerate(list2com):
                self.txData[-count-1] = int(elem)

        emitter.setChannel(self.agent_tag)
        message = struct.pack('@'+'i' * 10, int(self.txData[0]),int(self.txData[1]),int(self.txData[2]),
                              int(self.txData[3]),int(self.txData[4]),int(self.txData[5]),int(self.txData[6]),
                              int(self.txData[7]),int(self.txData[8]),int(self.txData[9]))
        emitter.send(message)

    def pre_webots_step(self, emitter):
        # \TODO remove the get function, shouldn't change from after_webots_step
        self.get_webots_location()
        self.set_webots_msg(pol=self.move_webots)
        self.send_webots_msg(emitter)

    def pre_webots_beacon_step(self, emitter):
        self.get_webots_location()
        self.set_webots_msg(pol=self.move_webots)
        self.send_webots_msg(emitter)

    def after_webots_step(self):
        self.get_webots_location()

    @staticmethod
    def normalize(item):
        # \TODO Build in that if norm of item is lower than threshold, return zero vector
        if np.linalg.norm(item) < step_threshold:
            return np.array([0.,0.])
        else:
            return item / np.linalg.norm(item)

    @staticmethod
    def cart2pol(cart=np.array([0, 0])):
        rho = np.sqrt(cart[0] ** 2 + cart[1] ** 2)
        phi = np.arctan2(cart[1], cart[0])
        return (np.array([rho, phi]))

    @staticmethod
    def pol2cart(pol=np.array([1,0])):
        x = pol[0] * np.cos(pol[1])
        y = pol[0] * np.sin(pol[1])
        return(np.array([x,y]))

    def neigh_agents(self, agents):
        self.neigh = [tag for tag in agents if np.linalg.norm(agents[tag].pt[1]
                                                - self.pt[1]) < agents[tag].range]
        self.neigh_foragers = [tag for tag in self.neigh if agents[tag].state[1] == 1]
        self.neigh_beacons = [tag for tag in self.neigh if agents[tag].state[1] == 0]

        if not numeric_step_margin:
            self.neigh_toll = self.neigh
            self.neigh_foragers_toll = self.neigh_foragers
            self.neigh_beacons_toll = self.neigh_beacons
        else:
            self.neigh_toll = [tag for tag in agents if np.linalg.norm(agents[tag].pt[1]
                                                - self.pt[1]) < agents[tag].range + numeric_step_margin]
            self.neigh_foragers = [tag for tag in self.neigh_toll if agents[tag].state[1] == 1]
            self.neigh_beacons = [tag for tag in self.neigh_toll if agents[tag].state[1] == 0]

    def adapt_range(self):
        if adapt_range_option == 'angle':
            # if sum(self.wv[0]) != 0. and sum(self.wv[1]) != 0.:
            if np.linalg.norm(self.v[0]) > step_threshold and np.linalg.norm(self.v[1]) > step_threshold:
                # /todo check is this is correct
                angle = angle_vectors(self.v[0], self.v[1])
                coefficient = (clip_range-min_clip_range)/np.pi
                self.range = coefficient*angle + min_clip_range
        elif adapt_range_option == 'weights':
            if self.w[1] == 0.:
                self.range = clip_range
            elif self.w[0] > 0:
                coefficient = clip_range - min_clip_range
                self.range = -coefficient*min(1,self.w[1]/self.w[0]) + clip_range
            else:
                self.range = min_clip_range
        elif adapt_range_option == 'no adaption':
            self.range = clip_range


    def _search_food(self):
        self.mode[0] = self.mode[1]
        self.mode[1] = 0

    def _search_nest(self):
        self.mode[0] = self.mode[1]
        self.mode[1] = 1

    def in_range(self, location):
        location = np.array(location)
        return np.linalg.norm(self.pt[1] - location) <= target_range + numeric_step_margin

    # def _pick_direction(self,beacons,ants):
    def move_random(self):
        # vec = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        # #\TODO we normalized the previous move, since this is generally smaller because of collision avoidance, \
        # #\TODO Consider to scale the stepts such that the old one is weighted more
        # return self.normalize(1*self.normalize(vec) * dt + 1.1*self.normalize(self.move[1])*dt) * dt

        move_polar = self.cart2pol(self.normalize(self.move[1]))
        random_move = self.pol2cart(np.array([1, move_polar[1] + np.random.normal(loc=0.0, scale=0.25, size=None)]))
        return self.normalize(random_move) * dt

    def _pick_direction(self, agents):
        if self.mode[1] == 0:
            w_type = 1
        else:
            w_type = 0
        self.move[0] = self.move[1]

        if self.mode[0] == 0 and self.in_range(food_location):
            return -self.normalize(self.move[1])*dt
        elif self.mode[1] == 1 and self.in_range(nest_location):
            return -self.normalize(self.move[1]) * dt


        weighted_vecs = [
            (agents[beac_tag].v[w_type] * np.array(
                    [np.random.normal(loc=dist_v_mean, scale=dist_v_std),
                     np.random.normal(loc=dist_v_mean, scale=dist_v_std)]))*(
                        agents[beac_tag].w[w_type] * abs(np.random.normal(loc=dist_w_mean, scale=dist_w_std)))
            for beac_tag in self.neigh_beacons_toll]

        weights = [agents[beac_tag].w[w_type] * abs(np.random.normal(loc=dist_w_mean, scale=dist_w_std)) for beac_tag
                   in self.neigh_beacons_toll]

        # check exceptions for not following the stored weighted vectors
        if not weighted_vecs:
            return self.move_random()
        elif max(weights) < step_threshold:
            return self.move_random()
        # elif np.isnan(weighted_vecs).any():
        #     return self.move_random()
        elif exploration_rate > random.uniform(0, 1):
            return self.move_random()
        # else:
        #     print('we moved non random')

        weighted_vecs = weighted_vecs/max(weights)

        # based on the pick option selection, pick direction based on stored weighted vectors
        if pick_option == 'max':
            max_index = np.argmax([np.linalg.norm(item) for item in weighted_vecs], axis=0)
            vec = weighted_vecs[max_index]
        elif pick_option == 'max_element_wise':
            vec = np.amax(weighted_vecs, axis=0)
        elif pick_option == 'average':
            vec = np.mean(weighted_vecs,axis=0)
        elif pick_option == 'sum':
            vec = sum(weighted_vecs)
        else:
            raise ValueError('ERROR - select pick option')

        vec = self.normalize(vec) * dt
        if np.isnan(vec).any() or np.linalg.norm(vec) <= step_threshold:
            # raise ValueError('generate vec in _pick direction contains NaN')
            print('generate vec in _pick direction contains NaN')
            return self.move_random()

        return vec

    def pre_sim_step(self,agents):
        self.pt[0] = self.pt[1]
        move_output = self._pick_direction(agents)
        print('robot: ' + str(self.agent_tag) + ' move: ' + str(move_output))

        if np.isnan(move_output).any() or np.linalg.norm(move_output) <= step_threshold:
            print('stop')

        self.move_webots = self.cart2pol(move_output)

    def pre_sim_beacon_step(self):
        self.pt[0] = self.pt[1]
        if beacons_move:
            try:
                if self.w[0] > step_threshold and self.w[1] > step_threshold and np.linalg.norm(
                        self.v[0]) > step_threshold and np.linalg.norm(self.v[1]) > step_threshold:
                    vec1 = self.normalize(self.v[0])
                    vec2 = self.normalize(self.v[1])
                    if not np.isnan(vec1).any() and not np.isnan(vec2).any():
                        move_output = (vec1 + vec2) * dt * rel_beac_speed
                        print('beacon move')
                        self.move_webots = self.cart2pol(move_output)
                        self.moving_beacon = True
                    else:
                        self.move_webots = np.array([0., 0.])
                else:
                    self.move_webots = np.array([0., 0.])
            except:
                self.move_webots = np.array([0.,0.])
        else:
            self.move_webots = np.array([0., 0.])

    def after_sim_step(self):
        self.pt[1] = self.pt_webots

        # self.pt[1], self.obs_avoid_mode = grid.obstacle_avoidance(self.pt[1], move_output)

        self.move[1] = self.pt[1] - self.pt[0]

        if np.isnan(self.pt).any():
            raise ValueError('generate vec in _pick direction contains NaN')
        if self.mode[1] == 0 and self.in_range(food_location):
            self.trips += 1
            self._search_nest()
        elif self.mode[1] == 1 and self.in_range(nest_location):
            self.trips += 1
            self._search_food()
        else:
            self.mode[0] = self.mode[1]

    def after_sim_beacon_step(self):
        self.pt[1] = self.pt_webots

        self.move[1] = self.pt[1] - self.pt[0]

        if np.isnan(self.pt).any():
            raise ValueError('generate vec in _pick direction contains NaN')
        if self.mode[1] == 0 and self.in_range(food_location):
            self._search_nest()
        elif self.mode[1] == 1 and self.in_range(nest_location):
            self._search_food()
        else:
            self.mode[0] = self.mode[1]


class Agents:
    def __init__(self, grid):
        self.agents = dict()
        self.grid = grid

        self.beac_tags = []
        self.forager_tags = []

    def find_neighs(self):
        for tag in self.agents:
            self.agents[tag].neigh_agents(self.agents)

    def check_states(self):
        self.beac_tags = []
        self.forager_tags = []
        for tag in self.agents:
            if self.agents[tag].state[1] == 0:
                self.beac_tags += [tag]
            else:
                self.forager_tags += [tag]

    def check_weights(self,to_check = 'W', thres=0.):
        if to_check == 'W1':
            return {beac_tag: self.agents[beac_tag].w[0] for beac_tag in self.beac_tags if
                    self.agents[beac_tag].w[0] > thres}
        elif to_check == 'W2':
            return {beac_tag: self.agents[beac_tag].w[1] for beac_tag in self.beac_tags if
                    self.agents[beac_tag].w[1] > thres}
        elif to_check == 'W':
            return {beac_tag: self.agents[beac_tag].w[0] + self.agents[beac_tag].w[1]
                    for beac_tag in self.beac_tags if self.agents[beac_tag].w[0] > thres or
                    self.agents[beac_tag].w[1] > thres}

    def check_foragers(self, thres=0):
        return {beac_tag: len(self.agents[beac_tag].neigh_foragers_toll) for beac_tag in self.beac_tags if
                len(self.agents[beac_tag].neigh_foragers_toll) > thres}

    def adapt_ranges(self):
        for beac_tag in self.beac_tags:
            self.agents[beac_tag].adapt_range()

    def pre_steps(self,emitter):
        for forager_tag in self.forager_tags:
            self.agents[forager_tag].pre_sim_step(self.agents)
            self.agents[forager_tag].pre_webots_step(emitter)

        for beacon_tag in self.beac_tags:
            self.agents[beacon_tag].pre_sim_beacon_step()
            self.agents[beacon_tag].pre_webots_beacon_step(emitter)

    def after_steps(self):
        for forager_tag in self.forager_tags:
            self.agents[forager_tag].after_webots_step()
            self.agents[forager_tag].after_sim_step()

        for beacon_tag in self.beac_tags:
            self.agents[beacon_tag].after_webots_step()
            self.agents[beacon_tag].after_sim_beacon_step()

    def get_max_webot_tdelta(self):
        waiting_times = [self.agents[tag].waiting_time for tag in self.agents]
        return max(waiting_times, default=0)

    def release_foragers(self,n_to_release,supervisor, t_webots,t_sim):
        # next_tag = max(self.agents.keys(),default=-1) + 1
        # release_tags = list(range(next_tag,next_tag+n_to_release))

        for tag in tags_release_order[t_sim]:
            self.agents[tag] = Agent(location = nest_location, agent_tag = tag,
                                     supervisor = supervisor, timestep = t_webots)
        return tags_release_order[t_sim]

    def switch_states(self):
        forager_tags_old = self.forager_tags.copy()
        beacons_tags_old = self.beac_tags.copy()
        tags_changed = []

        if not move_random:
            for tag in forager_tags_old:
                self.agents[tag].state[0] = self.agents[tag].state[1]

                if not self.agents[tag].neigh_beacons:
                    tags_changed += [tag]
                    self.agents[tag].state[1] = 0

                    # /todo check if we need to drop the initialization of vw
                    self.update_beacon_w_v(tag,[tag])

                    # update_agents in efficient way, adapting range not neces. Alternatively use update_agents
                    self.beac_tags += [tag]
                    self.forager_tags.remove(tag)
                    self.find_neighs()

            weights_check = self.check_weights(thres=threshold)
            foragers_check = self.check_foragers()
            #
            # for tag in beacons_tags_old:
            #     self.agents[tag].state[0] = self.agents[tag].state[1]
            #
            #     if tag not in weights_check and tag not in foragers_check and tag not in tags_changed:
            #         self.agents[tag].state[1] = 1
            #
            #         # update_agents in efficient way, adapting range not neces. Alternatively use update_agents
            #         self.beac_tags.remove(tag)
            #         self.forager_tags += [tag]
            #         self.find_neighs()
            #
            #         # In theory the foragers_check can change when an beacon is remove, so update foragers_check
            #         foragers_check = self.check_foragers()
            #
            #     elif len(self.agents[tag].neigh_beacons) > 1:
            #         neigh_w0 = [self.agents[tag_beac].w[0] for tag_beac in self.agents[tag].neigh_beacons if
            #                     tag_beac != tag]
            #         neigh_w1 = [self.agents[tag_beac].w[1] for tag_beac in self.agents[tag].neigh_beacons if
            #                     tag_beac != tag]
            #         if max(neigh_w0, default=0) > self.agents[tag].w[0] and max(neigh_w1, default=0) > \
            #                 self.agents[tag].w[1]:
            #             self.agents[tag].state[1] = 1
            #
            #             # update_agents in efficient way, adapting range not neces. Alternatively use update_agents
            #             self.beac_tags.remove(tag)
            #             self.forager_tags += [tag]
            #             self.find_neighs()
            #
            #             # In theory the foragers_check can change when an beacon is remove, so update foragers_check
            #             foragers_check = self.check_foragers()


    def update_weights(self):
        # /TODO We now reward for every ant that finds the food! Not in line with our concept

        # update foraging agents
        for forager_tag in self.forager_tags:
            self.update_forager_w_v(forager_tag)

        # update beacon agents
        for beacon_tag in self.beac_tags:
            self.update_beacon_w_v(beacon_tag, self.agents[beacon_tag].neigh_foragers)

    def update_forager_w_v(self, forager_tag):
        for w_type in [0,1]:
            self.agents[forager_tag].v[w_type] = np.array([0.,0.])
            self.agents[forager_tag].w[w_type] = 0.

    def reward(self, weights, rew):
        # return rho * (lam * max(weights, default=0) + rew)
        if rew:
            return rho*rew
        else:
            return rho * (lam * max(weights, default=0))

    def update_beacon_w_v(self,beac_tag, forager_tags):
        beac_v = np.array([[0., 0.], [0., 0.]])
        foragers_w_update = np.array([0., 0.])
        count_v0 = 0
        count_v1 = 0

        # Introduce listen cap for beacons
        if beac_list_cap <= len(forager_tags):
            sample_forager_tags = np.random.choice(forager_tags, beac_list_cap, replace=False)
        else:
            sample_forager_tags = forager_tags

        for forager_tag in sample_forager_tags:
            W1_weights = [self.agents[tag].w[0] * abs(np.random.normal(loc=dist_w_mean, scale=dist_w_std)) for tag in
                          self.agents[forager_tag].neigh_beacons]
            W2_weights = [self.agents[tag].w[1] * abs(np.random.normal(loc=dist_w_mean, scale=dist_w_std)) for tag in
                          self.agents[forager_tag].neigh_beacons]

            forager_move = self.agents[forager_tag].move[1] * np.array(
                [np.random.normal(loc=dist_v_mean, scale=dist_v_std),
                 np.random.normal(loc=dist_v_mean, scale=dist_v_std)])

            forager_w_update = np.array([0., 0.])

            # if self.agents[forager_tag].mode[0] == 0:
            #     if self.agents[forager_tag].in_range(nest_location):
            #         forager_w_update[0] = self.reward(W1_weights, rew)
            #
            #         count_v0 += 1
            #         if use_weights_updating_v:
            #             beac_v[0] += -forager_move * forager_w_update[0]
            #         else:
            #             beac_v[0] += -forager_move
            #     else:
            #         forager_w_update[0] = self.reward(W1_weights, 0, )
            #
            #         count_v0 += 1
            #         if use_weights_updating_v:
            #             beac_v[0] += -forager_move * forager_w_update[0]
            #         else:
            #             beac_v[0] += -forager_move
            #
            #     if self.agents[forager_tag].in_range(food_location):
            #         forager_w_update[1] = self.reward(W2_weights, rew)
            #
            #         # count_v1 += 1
            #         # if use_weights_updating_v:
            #         #     beac_v[1] += forager_move * forager_w_update[1]
            #         # else:
            #         #     beac_v[1] += forager_move
            #     else:
            #         forager_w_update[1] = self.reward(W2_weights, 0, )
            #
            #         count_v1 += 1
            #         if use_weights_updating_v:
            #             beac_v[1] += forager_move * forager_w_update[1]
            #         else:
            #             beac_v[1] += forager_move
            #
            # elif self.agents[forager_tag].mode[0] == 1:
            #     if self.agents[forager_tag].in_range(food_location):
            #         forager_w_update[1] = self.reward(W2_weights, rew)
            #
            #         count_v1 += 1
            #         if use_weights_updating_v:
            #             beac_v[1] += -forager_move * forager_w_update[1]
            #         else:
            #             beac_v[1] += -forager_move
            #     else:
            #         forager_w_update[1] = self.reward(W2_weights, 0, )
            #
            #         count_v1 += 1
            #         if use_weights_updating_v:
            #             beac_v[1] += -forager_move * forager_w_update[1]
            #         else:
            #             beac_v[1] += -forager_move
            #
            #     if self.agents[forager_tag].in_range(nest_location):
            #         forager_w_update[0] = self.reward(W1_weights, rew)
            #
            #         # count_v0 += 1
            #         # if use_weights_updating_v:
            #         #     beac_v[0] += forager_move * forager_w_update[0]
            #         # else:
            #         #     beac_v[0] += forager_move
            #     else:
            #         forager_w_update[0] = self.reward(W1_weights, 0, )
            #
            #         count_v0 += 1
            #         if use_weights_updating_v:
            #             beac_v[0] += forager_move * forager_w_update[0]
            #         else:
            #             beac_v[0] += forager_move


            if self.agents[forager_tag].mode[0] == 0:
                if self.agents[forager_tag].in_range(nest_location):
                    forager_w_update[0] = self.reward(W1_weights, rew)

                    count_v0 += 1
                    if use_weights_updating_v:
                        beac_v[0] += -forager_move * forager_w_update[0]
                    else:
                        beac_v[0] += -forager_move

                elif self.agents[forager_tag].in_range(food_location):
                    forager_w_update[1] = self.reward(W2_weights, rew)

                else:
                    forager_w_update[0] = self.reward(W1_weights, 0, )

                    count_v0 += 1
                    if use_weights_updating_v:
                        beac_v[0] += -forager_move * forager_w_update[0]
                    else:
                        beac_v[0] += -forager_move

            elif self.agents[forager_tag].mode[0] == 1:
                if self.agents[forager_tag].in_range(food_location):
                    forager_w_update[1] = self.reward(W2_weights, rew)

                    count_v1 += 1
                    if use_weights_updating_v:
                        beac_v[1] += -forager_move * forager_w_update[1]
                    else:
                        beac_v[1] += -forager_move

                elif self.agents[forager_tag].in_range(nest_location):
                    forager_w_update[0] = self.reward(W1_weights, rew)

                else:
                    forager_w_update[1] = self.reward(W2_weights, 0)

                    count_v1 += 1
                    if use_weights_updating_v:
                        beac_v[1] += -forager_move * forager_w_update[1]
                    else:
                        beac_v[1] += -forager_move

            foragers_w_update += forager_w_update
            self.agents[beac_tag].w[0] += forager_w_update[0] / (len(sample_forager_tags)) #+1 for self loop
            self.agents[beac_tag].w[1] += forager_w_update[1] / (len(sample_forager_tags)) #+1 for self loop

        # # Also update the beacon by itself
        # # /TODO the self-loop includes the weights just updated, better, just include the beac_tag in the tags of for-loop?
        # W1_weights = [self.agents[tag].w[0] for tag in self.agents[beac_tag].neigh_beacons]
        # W2_weights = [self.agents[tag].w[1] for tag in self.agents[beac_tag].neigh_beacons]
        #
        # if self.agents[beac_tag].in_range(nest_location):
        #     self.agents[beac_tag].w[0] += self.reward(W1_weights, rew) / (len(sample_forager_tags)+1) #+1 for self loop
        # else:
        #     self.agents[beac_tag].w[0] += self.reward(W1_weights, 0) / (len(sample_forager_tags)+1)  # +1 for self loop
        #
        # if self.agents[beac_tag].in_range(food_location):
        #     self.agents[beac_tag].w[1] += self.reward(W2_weights, rew) / (len(sample_forager_tags)+1) #+1 for self loop
        # else:
        #     self.agents[beac_tag].w[1] += self.reward(W2_weights, 0) / (len(sample_forager_tags)+1)  # +1 for self loop

        if np.linalg.norm(self.agents[beac_tag].v[0]) and count_v0:
            self.agents[beac_tag].v[0] *= (1 - rho_v)
            if use_weights_updating_v:
                self.agents[beac_tag].v[0] += rho_v * beac_v[0] / (count_v0 * foragers_w_update[0])
            else:
                self.agents[beac_tag].v[0] += rho_v * beac_v[0] / (count_v0)
        elif count_v0:
            if use_weights_updating_v:
                if use_rhov_2_init:
                    self.agents[beac_tag].v[0] += rho_v * beac_v[0] / (count_v0 * foragers_w_update[0])
                else:
                    self.agents[beac_tag].v[0] += beac_v[0] / (count_v0 * foragers_w_update[0])
            else:
                if use_rhov_2_init:
                    self.agents[beac_tag].v[0] += rho_v * beac_v[0] / (count_v0)
                else:
                    self.agents[beac_tag].v[0] += beac_v[0] / (count_v0)

        if np.linalg.norm(self.agents[beac_tag].v[1]) and count_v1:
            self.agents[beac_tag].v[1] *= (1 - rho_v)
            if use_weights_updating_v:
                self.agents[beac_tag].v[1] += rho_v * beac_v[1] / (count_v1 * foragers_w_update[1])
            else:
                self.agents[beac_tag].v[1] += rho_v * beac_v[1] / (count_v1)
        elif count_v1:
            if use_weights_updating_v:
                if use_rhov_2_init:
                    self.agents[beac_tag].v[1] += rho_v * beac_v[1] / (count_v1 * foragers_w_update[1])
                else:
                    self.agents[beac_tag].v[1] += beac_v[1] / (count_v1 * foragers_w_update[1])
            else:
                if use_rhov_2_init:
                    self.agents[beac_tag].v[1] += rho_v * beac_v[1] / (count_v1)
                else:
                    self.agents[beac_tag].v[1] += beac_v[1] / (count_v1)

        if self.agents[beac_tag].in_range(nest_location):
            self.agents[beac_tag].v[0] = np.array([0., 0.])
        if self.agents[beac_tag].in_range(food_location):
            self.agents[beac_tag].v[1] = np.array([0., 0.])

    def evaporate_weights(self):
        for beac_tag in self.beac_tags:
            self.agents[beac_tag].w *= (1-rho)

    def load_states(self, t_webots, supervisor, load_time):
        if obstacles:
            obs_text = 'True'
        else:
            obs_text = 'False'

        with open('./saved_webots_states/saved_end_data_A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMoveFalse_cap100_stdw0_stdv0_RUN1.p'.format(
                N_total,total_time,obs_text,rho,exploration_rate,clip_range, super_tag, load_time), 'rb') as fp:
            saved_states = pickle.load(fp)

        for tag in saved_states:
            self.agents[tag] = Agent(location=nest_location, agent_tag=tag,
                                     supervisor=supervisor, timestep=t_webots)
            self.agents[tag].pt = saved_states[tag]['pt']
            self.agents[tag].w = saved_states[tag]['w']
            self.agents[tag].v = saved_states[tag]['v']
            self.agents[tag].move = saved_states[tag]['move']
            self.agents[tag].mode = saved_states[tag]['mode']
            self.agents[tag].state = saved_states[tag]['state']
            self.agents[tag].trips = saved_states[tag]['trips']
            self.agents[tag].neigh = saved_states[tag]['neigh']
            self.agents[tag].neigh_foragers = saved_states[tag]['neigh_foragers']
            self.agents[tag].neigh_beacons = saved_states[tag]['neigh_beacons']
            self.agents[tag].neigh_toll = saved_states[tag]['neigh_toll']
            self.agents[tag].neigh_foragers_toll = saved_states[tag]['neigh_foragers_toll']
            self.agents[tag].neigh_beacons_toll = saved_states[tag]['neigh_beacons_toll']
            self.agents[tag].range = saved_states[tag]['range']
