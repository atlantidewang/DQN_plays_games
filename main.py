import sys
import dqn
sys.path.append("./flappy_bird_game/")
sys.path.append("./shoot_plane_game/")
import wrapped_flappy_bird
import flappy_bird_conf
import shoot_plane
import shoot_plane_conf

# flappy bird training
#dqn.DQN_Trainer(wrapped_flappy_bird.GameState(), shoot_plane_conf).train_dqn()

# shoot plane training
dqn.DQN_Trainer(shoot_plane.ShootPlaneGame(), shoot_plane_conf).train_dqn()
