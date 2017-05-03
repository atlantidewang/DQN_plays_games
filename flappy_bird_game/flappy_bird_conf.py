# flappy bird configuration
game_name = 'flappy_bird' # the name of the game being played for log files
num_actions = 2 # number of valid actions
gamma = 0.99 # decay rate of past observations
num_observations = 10000. # timesteps to observe before training
num_explorations = 2000000. # frames over which to anneal epsilon
final_epsilon = 0.01 # final value of epsilon
init_epsilon = 0.6 # starting value of epsilon
size_replay_mem = 10000 # number of previous transitions to remember
size_batch = 32 # size of minibatch
frames_per_action = 3

# cnn parameters
in_shape = 		[80, 80]
in_channel = 	4
filters = 		[[8, 8], [4, 4], [3, 3]]
out_channels = 	[32, 64, 64]
strides = 		[4, 2, 1]
#if pool = False, then should notice that size do not shrink and should adjust the paramters.
pool=			True

# bp parameters
cnn_out_hidden_size = 256
full_connected_sizes = [256, num_actions]

# optimizer parameter
adam_learning_rate = 1e-5

# picture process parameters
thresh = 1
max_value = 255
size_pic_stack = 4 