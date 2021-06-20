from configuration import *
from domain import *
from agents import *

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d
import pickle
import copy
import matplotlib.patches as patches

if local:
    FOLDER_LOCATION = './figures/'
else:
    FOLDER_LOCATION = './figures_manuels_desk/'


class Simulations:
    def __init__(self, supervisor, t_webots, t_sim, saved_states=False):
        self.grid = Grid(obstacles=obstacles)
        self.total_trips_abs = dict()
        self.total_trips_rel = dict()

        self.agents = Agents(self.grid)

        if not saved_states:
            _ = self.agents.release_foragers(1, supervisor, t_webots, t_sim)
        else:
            self.agents.load_states(t_webots=t_webots, supervisor=supervisor)

        self.update_agents()

        self.agents.switch_states()

        self.saved_data = {}

        # In theory we do not have to update the agents again, since we updated all info in the switch_state function
        # self.update_agents()

    def sim_pre_step(self, time_step, emitter, supervisor, timestep):
        # N_till_now = (time_step+1)*N_batch
        N_till_now = len(self.agents.beac_tags) + len(self.agents.forager_tags)

        # ACTION
        if N_till_now < N_total:
            release_tags = self.agents.release_foragers(N_batch, supervisor, t_webots=timestep, t_sim=time_step)
            # \todo remove beacon update, it isn't even sure if this are the only foragers within its range
            self.agents.update_beacon_w_v(1, release_tags)

        # UPDATE
        self.update_agents()

        # ACTION
        self.agents.pre_steps(emitter)

    def sim_after_step(self, time_step, switch_time=250):
        # ACTION
        self.agents.after_steps()

        # UPDATE
        self.update_agents()

        # ACTION
        if time_step >= switch_time:
            self.agents.switch_states()

        # UPDATE
        ## Done within switch_step

        # ACTION
        self.agents.evaporate_weights()
        self.agents.update_weights()

        # UPDATE
        self.update_agents()

        # STORE
        self.store_nr_trips(time_step)
        self.store_data(time_step)

    def update_agents(self):
        self.agents.check_states()
        self.agents.adapt_ranges()  # /todo do we want to update the ranges all the time?
        self.agents.find_neighs()

    def store_nr_trips(self, t):
        trips = [self.agents.agents[tag].trips for tag in self.agents.agents]
        self.total_trips_abs[t] = sum(trips)
        self.total_trips_rel[t] = sum(trips) / len(self.agents.agents)

    def plt_only_weights_and_vectors(self, to_plot='W', fig_tag=None):
        # if len(self.agents.beac_tags) > 3:
        #     vor = Voronoi([np.array([self.agents.agents[beac_tag].pt[1][0],
        #                              -self.agents.agents[beac_tag].pt[1][1]]) for beac_tag in self.agents.beac_tags])
        #     voronoi_plot_2d(vor, show_vertices=False)

        ax = plt.gca()
        for item in obstacles:
            rect = patches.Rectangle((obstacles[item][0][0], -obstacles[item][1][1]), 2, 2, linewidth=1, fill=True,
                                     zorder=1)
            ax.add_patch(rect)


        # /TODO now plotting only vectors larger than threshold, what we want?
        if to_plot == 'W1':
            checked_ws = self.agents.check_weights(to_check='W1', thres=step_threshold)

            for beac_tag in checked_ws:
                size = np.log(checked_ws[beac_tag] / max(checked_ws.values()) + 1) * 10
                plt.plot([self.agents.agents[beac_tag].pt[1][0]], [-self.agents.agents[beac_tag].pt[1][1]],
                         'o', color='black', markersize=size)

                if np.linalg.norm(self.agents.agents[beac_tag].v[0]) > step_threshold:
                    arrow = self.normalize(self.agents.agents[beac_tag].v[0]) * dt
                    plt.plot([self.agents.agents[beac_tag].pt[1][0], self.agents.agents[beac_tag].pt[1][0] + arrow[0]],
                             [-self.agents.agents[beac_tag].pt[1][1],
                              -(self.agents.agents[beac_tag].pt[1][1] + arrow[1])], color='black')


        elif to_plot == 'W2':
            checked_ws = self.agents.check_weights(to_check='W2', thres=step_threshold)

            for beac_tag in checked_ws:
                size = np.log(checked_ws[beac_tag] / max(checked_ws.values()) + 1) * 10
                plt.plot([self.agents.agents[beac_tag].pt[1][0]], [-self.agents.agents[beac_tag].pt[1][1]],
                         'o', color='black', markersize=size)

                if np.linalg.norm(self.agents.agents[beac_tag].v[1]) > step_threshold:
                    arrow = self.normalize(self.agents.agents[beac_tag].v[1]) * dt
                    plt.plot([self.agents.agents[beac_tag].pt[1][0], self.agents.agents[beac_tag].pt[1][0] + arrow[0]],
                             [-self.agents.agents[beac_tag].pt[1][1],
                              -(self.agents.agents[beac_tag].pt[1][1] + arrow[1])],
                             color='black')

        elif to_plot == 'W':
            checked_w0s = self.agents.check_weights(to_check='W1', thres=step_threshold)
            checked_w1s = self.agents.check_weights(to_check='W2', thres=step_threshold)

            checked_ws = self.agents.check_weights(to_check='W', thres=step_threshold)

            for beac_tag in checked_ws:
                size = np.log(checked_ws[beac_tag] / max(checked_ws.values()) + 1) * 10
                if np.isnan(size):
                    print('check hier')
                plt.plot([self.agents.agents[beac_tag].pt[1][0]], [-self.agents.agents[beac_tag].pt[1][1]],
                         'o', color='black', markersize=size)

                if beac_tag in checked_w0s and np.linalg.norm(self.agents.agents[beac_tag].v[0]) > step_threshold:
                    arrow0 = self.normalize(self.agents.agents[beac_tag].v[0]) * dt
                    plt.plot([self.agents.agents[beac_tag].pt[1][0], self.agents.agents[beac_tag].pt[1][0] + arrow0[0]],
                             [-self.agents.agents[beac_tag].pt[1][1],
                              -(self.agents.agents[beac_tag].pt[1][1] + arrow0[1])], color='black')
                if beac_tag in checked_w1s and np.linalg.norm(self.agents.agents[beac_tag].v[1]) > step_threshold:
                    arrow1 = self.normalize(self.agents.agents[beac_tag].v[1]) * dt
                    plt.plot([self.agents.agents[beac_tag].pt[1][0], self.agents.agents[beac_tag].pt[1][0] + arrow1[0]],
                             [-self.agents.agents[beac_tag].pt[1][1],
                              -(self.agents.agents[beac_tag].pt[1][1] + arrow1[1])], color='blue')

        plt.plot([nest_location[0], food_location[0]],
                 [-nest_location[1], -food_location[1]], 'r*')

        plt.plot([self.agents.agents[forager_tag].pt[1][0] for forager_tag in self.agents.forager_tags if
                  self.agents.agents[forager_tag].mode[1] == 0],
                 [-self.agents.agents[forager_tag].pt[1][1] for forager_tag in self.agents.forager_tags if
                  self.agents.agents[forager_tag].mode[1] == 0], 'g*', markersize=2)
        plt.plot([self.agents.agents[forager_tag].pt[1][0] for forager_tag in self.agents.forager_tags if
                  self.agents.agents[forager_tag].mode[1] == 1],
                 [-self.agents.agents[forager_tag].pt[1][1] for forager_tag in self.agents.forager_tags if
                  self.agents.agents[forager_tag].mode[1] == 1], 'y*', markersize=2)

        # plt.xlim(domain[0][0] - 1, domain[0][1] + 1)
        # plt.ylim(domain[1][0] - 1, domain[1][1] + 1)
        plt.xlim(domain[0][0], domain[2][0])
        plt.ylim(-domain[0][1], -domain[1][1])

        # for line in self.grid.ref_lines:
        #     plt.plot([item[0] for item in line], [item[1] for item in line], 'r')

        # plt.colorbar()
        if to_plot == 'W1' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W1_WEIGHTS/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W2' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W2_WEIGHTS/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W_WEIGHTS/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_trips(self, t, run, save=True):
        trips_sequence = np.array([self.total_trips_abs[time] for time in range(0, t)]) / N_total

        plt.plot(np.array(range(0, t)) * dt, trips_sequence, 'r')
        plt.xlabel("Time")
        plt.ylabel("#Trips / #Agents")

        if obstacles:
            obs_text = 'True'
        else:
            obs_text = 'False'

        if save:
            plt.savefig(
                FOLDER_LOCATION + 'total_trips__A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMove{}_cap{}_stdw{}_stdv{}_RUN{}.png'.format(
                    N_total,
                    t,
                    obs_text,
                    rho,
                    exploration_rate,
                    clip_range,
                    super_tag,
                    total_time, beacons_move,
                    beac_list_cap, dist_w_std, dist_v_std,
                    run))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def normalize(item):
        return_value = item / np.linalg.norm(item)
        if np.isnan(return_value).any():
            print('return value in to normalize value')
            return None
        else:
            return return_value

    def store_data(self, t):
        self.saved_data[t] = dict()
        for tag in self.agents.forager_tags + self.agents.beac_tags:
            self.saved_data[t][tag] = {'pt': copy.deepcopy(self.agents.agents[tag].pt),
                                       'w': copy.deepcopy(self.agents.agents[tag].w),
                                       'v': copy.deepcopy(self.agents.agents[tag].v),
                                       'mode': copy.deepcopy(self.agents.agents[tag].mode),
                                       'state': copy.deepcopy(self.agents.agents[tag].state),
                                       'trips': copy.deepcopy(self.agents.agents[tag].trips)}

    def save_data(self, t, run):
        # Save data per time step
        with open('saved_data_A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMove{}_cap{}_stdw{}_stdv{}_RUN{}.p'.format(
                N_total, total_time, obs_text,
                rho, exploration_rate,
                clip_range, super_tag, t, beacons_move,
                beac_list_cap, dist_w_std, dist_v_std, run),
                'wb') as fp:
            pickle.dump(self.saved_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # Save end data
        saved_states = dict()
        for tag in self.agents.forager_tags + self.agents.beac_tags:
            saved_states[tag] = {'pt': self.agents.agents[tag].pt,
                                 'w': self.agents.agents[tag].w,
                                 'v': self.agents.agents[tag].v,
                                 'move': self.agents.agents[tag].move,
                                 'mode': self.agents.agents[tag].mode,
                                 'state': self.agents.agents[tag].state,
                                 'trips': self.agents.agents[tag].trips,
                                 'neigh': self.agents.agents[tag].neigh,
                                 'neigh_foragers': self.agents.agents[tag].neigh_foragers,
                                 'neigh_beacons': self.agents.agents[tag].neigh_beacons,
                                 'neigh_toll': self.agents.agents[tag].neigh_toll,
                                 'neigh_foragers_toll': self.agents.agents[tag].neigh_foragers_toll,
                                 'neigh_beacons_toll': self.agents.agents[tag].neigh_beacons_toll,
                                 'range': self.agents.agents[tag].range,
                                 'sim_time': t}

            with open(
                    'saved_end_data_A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMove{}_cap{}_stdw{}_stdv{}_RUN{}.p'.format(
                            N_total, total_time,
                            obs_text,
                            rho, exploration_rate,
                            clip_range, super_tag,
                            t, beacons_move,
                            beac_list_cap, dist_w_std, dist_v_std, run), 'wb') as fp:
                pickle.dump(saved_states, fp, protocol=pickle.HIGHEST_PROTOCOL)
