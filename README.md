## python dependencies:

  * TensorFlow 1.0
  * pygame
  * OpenCV-Python

## references: 
  * https://github.com/yenchenlin/DeepLearningFlappyBird
  * https://github.com/libratears/PyShootGame
  
## how to add new games to play:
  Edit the *_conf.py in the coresponding game's directory. Edit you game's code to be a class with a member method "frame_step(self, input_actions)", which returns a tuple(image_data, reward, terminal). Then add the configuration and game class instance into "dqn.DQN_Trainer("instance", "conf").train_dqn()". 
## how to run:
 * run the flappy bird: python main.py bird
 * run the shoot plane: python main.py palne
 * clean the saved networks and run the code: ./clean_run.sh bird or ./clean_run.sh plane
