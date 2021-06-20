save_steps = True
start_run = 1
total_runs = 1
super_tag = 60
beacons_move = False
rel_beac_speed = 0.05
move_random = False

local = False
total_time = 300  # 200
dt = 0.75
WEBOTS_BASIC_TIME = 64

beac_list_cap = 100
dist_w_mean = 1
dist_w_std = 0
dist_v_mean = 1
dist_v_std = 0

# /TODO if using non sym obstacle, change plotting (add mirroring)
# domain = [[-1.5, 1.5], [-1.25, 1.25]]
domain = [[-3,2],[-3,-6],[7,-6],[7,2]]

# provide corner points of as: obstacle = [lower left, upper left, upper right, lower right]
# obstacle = [[-0.3, -0.45], [-0.3, 0.45], [0.3, 0.45], [0.3, -0.45]]
obstacles = {1: [[-1,-4], [-1,-2],[1,-2],[1,-4]],
            2: [[3,-2],[3,0],[5,0],[5,-2]]}
# obstacle = None

nest_location = [-1., 0]  # [5.,4.] / [5., 4.]
food_location = [4.5, -4]  # [30.,14.] / [15., 6.]
default_beacon_grid = [10, 8]  # [20,16] / [10,8]

N_batch = 2
N_total = 101 # 500 / 100

kappa = 1  # 1
lam = 0.8  # 1
rew = 1
rho = 0.01  # 0.0001
rho_v = 0.001
default_epsilon = 0.01  # 0.05 #5
exploration_rate = 0.01  # previously called default_epsilon
DEBUG = 1
target_range = 0.8
# default_var = 10
clip_range = 1.2# 1.  # ,, 2.
min_clip_range = 1.2 #1.
threshold = 1e-6
step_threshold = 1e-6  # 1e-3   # 1e-7
move_type = 'add_switch'  # 'der'/ 'add' / 'add_switch'
numeric_step_margin = 0
use_weights_updating_v = False
use_rhov_2_init = True
adapt_range_option = 'weights'  # weights, angle, no adaption
pick_option = 'sum'  # 'max 'max_element_wise' , 'average'

# Prepare variables
if obstacles:
    obs_text = 'True'
else:
    obs_text = 'False'

tags_release_order = {-1: [1]}
tags2release = list(range(1, N_total+1))[::-1]
for i in range(0, int((N_total - 1) / N_batch)):
    tags_release_order[i] = tags2release[((i) * 2):((i + 1) * 2)]
