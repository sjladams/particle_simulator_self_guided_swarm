"""weighted_landmark_supervisor controller."""

from controller import Supervisor
from simulation import *
import sys
import time

# run = int(sys.argv[1])
run = 1

# Set up Webots stuff
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
emitter = supervisor.getDevice('emitter')

# Set up Simulation stuff
simulation = Simulations(supervisor, t_webots=timestep, t_sim=-1)

load_time = 399
simulation.agents.load_states(t_webots=timestep, supervisor=supervisor, load_time=load_time)

for t in range(0, int(total_time / dt)):
    print(t)
    dumy = supervisor.step(timestep)

    timestep = int(supervisor.getBasicTimeStep())
    simulation.sim_pre_step(t, emitter, supervisor, timestep)

    max_tdelta = simulation.agents.get_max_webot_tdelta()

    for i in range(0, max_tdelta):
        dumy = supervisor.step(timestep)
        timestep = int(supervisor.getBasicTimeStep())

    simulation.sim_after_step(t, switch_time=0)

    if save_steps and t % 1 == 0:
        simulation.plt_only_weights_and_vectors(to_plot="W", fig_tag="time_{}".format(t))
        if t % 5 ==0:
            simulation.save_data(t, run)
            supervisor.worldSave(
                '/home/steven/webots/Elisa3_Webots_Foraging/worlds/SUPER{}/sim_A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMove{}_cap{}_stdw{}_stdv{}_RUN{}.wbt'.format(
                    super_tag, N_total, total_time, obs_text, rho, exploration_rate, clip_range, super_tag, t, beacons_move,
                    beac_list_cap, dist_w_std, dist_v_std, run))

simulation.plot_trips(int(t), run, save=True)
simulation.save_data(t, run)

supervisor.worldSave(
    '/home/steven/webots/Elisa3_Webots_Foraging/worlds/SUPER{}/sim_A{}_T{}_O{}_rho{}_E{}_range{}_SUPER{}_t{}_BeacMove{}_cap{}_stdw{}_stdv{}_RUN{}.wbt'.format(
        super_tag, N_total, total_time, obs_text, rho, exploration_rate, clip_range, super_tag, t, beacons_move,
        beac_list_cap, dist_w_std, dist_v_std, run))

# supervisor.simulationQuit(0)
time.sleep(60*60*3)
print('finished run {}'.format(run))
