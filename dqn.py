#!/usr/bin/env python
from __future__ import print_function
from collections import deque
import tensorflow as tf 
import numpy as np 
import random
import os

import cv2
import sys
# tf.logging.set_verbosity(tf.logging.INFO)


class CNN:
	def __init__(self, in_shape, in_channel, filters, out_channels, strides, pool=True):
		self.in_shape = in_shape
		self.in_channel = in_channel
		self.filters = filters
		self.out_channels = out_channels
		self.strides = strides
		self.pool = pool
		self.in_layer = []
		self.out_hidden = []
		self.readout = []

	def __weight_variable(self, shape):
		init = tf.truncated_normal(shape, stddev = 0.01) # avoid symmetry and overfitting. (abs(x) < 2*sigma)
		return tf.Variable(init)

	def __bias_variable(self, shape):
		init = tf.constant(0.01, shape = shape)
		return tf.Variable(init)

	def __conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def __max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def __create_cnn_hidden_layer(self, input, filter, in_channel, out_channel, stride, pool = False):
		W = self.__weight_variable(filter + [in_channel] + [out_channel])
		b = self.__bias_variable([out_channel])
		h_conv = tf.nn.relu(self.__conv2d(input, W, stride) + b)
		if pool == True:
			h_conv = self.__max_pool_2x2(h_conv)
		return h_conv

	def __create_bp_hidden_layer(self, input, in_size, hidden_size):
		W = self.__weight_variable([in_size] + [hidden_size])
		b = self.__bias_variable([hidden_size])
		return tf.nn.relu(tf.matmul(input, W) + b)

	def create_cnn(self, cnn_out_hidden_size, full_connected_sizes):
		# batch_size * rows * colums * channels
		self.in_layer = tf.placeholder(tf.float32, [None] + self.in_shape + [self.in_channel])

		# interatively create hidden layer
		self.out_hidden = self.__create_cnn_hidden_layer(self.in_layer, \
												self.filters[0], \
												self.in_channel, \
												self.out_channels[0], \
												self.strides[0], \
												self.pool)
		# not pooling
		for i in xrange(1, len(self.out_channels)):
			self.out_hidden = self.__create_cnn_hidden_layer(self.out_hidden, \
										self.filters[i], \
										self.out_channels[i -  1], \
										self.out_channels[i], \
										self.strides[i], \
										False)#self.pool)

		# create full connected backprogate neural networks.
		out_hidden_flat = tf.reshape(self.out_hidden, [-1, cnn_out_hidden_size])
		self.readout = self.__create_bp_hidden_layer(out_hidden_flat, \
												cnn_out_hidden_size, \
												full_connected_sizes[0])

		for i in xrange(1, len(full_connected_sizes)):
			self.readout = self.__create_bp_hidden_layer(self.readout, \
												full_connected_sizes[i - 1], \
												full_connected_sizes[i])


#===============================================================================
class DQN_Trainer:
	def __init__(self, game, game_conf):
		self.game_conf = game_conf
		self.cnn = CNN(self.game_conf.in_shape, \
						self.game_conf.in_channel, \
						self.game_conf.filters, \
						self.game_conf.out_channels, \
						self.game_conf.strides, \
						self.game_conf.pool)
		self.cnn.create_cnn(self.game_conf.cnn_out_hidden_size, self.game_conf.full_connected_sizes)
		self.trainer = None
		self.D = deque()
		self.epsilon = self.game_conf.init_epsilon
		self.t = 0
		self.game_state = game

		if not os.path.exists("logs/"):
			os.mkdir("logs/")
		if not os.path.exists("logs/" + self.game_conf.game_name + "/"):
			os.mkdir("logs/" + self.game_conf.game_name)
		if not os.path.exists("saved_networks/"):
			os.mkdir("saved_networks/")
		self.readout_file = open("logs/" + self.game_conf.game_name + "/readout.txt", "w")
		self.hidden_file = open("logs/" + self.game_conf.game_name + "/hidden.txt", "w")

		self.s_t = []
		self.s_t_1 = [] # state
		self.r_t = []
		self.r_t_1 = [] # reward
		self.q_i = None
		self.q_i_1 = None # Q(s, a), tensor
		self.action = np.zeros(self.game_conf.num_actions) # num_actions * 1
		self.action_index = 0 # which action is performed
		self.readout_t = None # Q(s, a) , real number: 1*2
		self.action_vec = None # batch * num_actions
		self.terminal_b = False
		self.deque_last = []

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()

	def train_dqn(self):
		self.__create_optimizer()

		# first state
		self.action.fill(0)
		self.action[0] = 1 # just one palce can be 1
		self.__get_first_game_state_reward()

		# start to explore and exploit
		while True:
			# choose an action epsilon greedily, and perform the action and oberve the reward and state
			self.__choose_actions()
			self.__get_next_game_state_reward()

			self.__update_epsilon()

			# store the transition in the replay memory D
			self.__store_transitions()

			# update cnn parameters
			self.__update_cnn_params()

			self.__print_info()

			self.s_t = self.s_t_1
			self.t += 1


	def validate_dqn(self):
		pass

	def __create_optimizer(self):
		self.action_vec = tf.placeholder(tf.float32, [None, self.game_conf.num_actions]) # previous action
		self.q_i_1 = tf.placeholder(tf.float32, [None]) # new Q(s, a)
		self.q_i = tf.reduce_sum(tf.multiply(self.cnn.readout, self.action_vec), axis = 1) # previous Q(s, a)
		cost = tf.reduce_mean(tf.square(self.q_i_1 - self.q_i))
		self.trainer = tf.train.AdamOptimizer(self.game_conf.adam_learning_rate).minimize(cost)
		self.sess.run(tf.global_variables_initializer())

		# we can start iterate from last place
		if not os.path.exists("saved_networks/" + self.game_conf.game_name + "/"):
			os.mkdir("saved_networks/" + self.game_conf.game_name + "/")
		check_point = tf.train.get_checkpoint_state("saved_networks/" + self.game_conf.game_name + "/")
		if check_point and check_point.model_checkpoint_path:
			self.saver.restore(self.sess, check_point.model_checkpoint_path)
			print("Successfully loaded:", check_point.model_checkpoint_path)
		else:
			print("Could not find old network weights")


	def __choose_actions(self):
		self.action.fill(0)
		self.readout_t = self.cnn.readout.eval(feed_dict = {self.cnn.in_layer: [self.s_t]})[0] # [[]]
		if self.t % self.game_conf.frames_per_action == 0:
			if random.random() <= self.epsilon:
				#print("-----------Random Action ---------------")
				self.action_index = random.randrange(self.game_conf.num_actions)
			else:
				self.action_index = np.argmax(self.readout_t)
		else:
			self.action_index = 0 # every four frames to perform an action
		self.action[self.action_index] = 1

	def __get_next_game_state_reward(self):
		img_t_1_color, self.r_t, self.terminal_b = self.game_state.frame_step(self.action)

		img_t_1 = cv2.cvtColor(cv2.resize(img_t_1_color, tuple(self.game_conf.in_shape)), \
									cv2.COLOR_BGR2GRAY)
		ret, img_t_1 = cv2.threshold(img_t_1, \
									self.game_conf.thresh, \
									self.game_conf.max_value, \
									cv2.THRESH_BINARY)
		img_t_1 = np.reshape(img_t_1, tuple(self.game_conf.in_shape + [1]))
		self.s_t_1 = np.append(img_t_1, \
								self.s_t[:, :, :self.game_conf.size_pic_stack - 1], \
								axis = 2) # now, prev1, prev2, prev3


	def __get_first_game_state_reward(self):
		img_t_color, self.r_t, self.terminal_b = self.game_state.frame_step(self.action)
		img_t = cv2.cvtColor(cv2.resize(img_t_color, tuple(self.game_conf.in_shape)), \
								cv2.COLOR_BGR2GRAY)
		ret, img_t = cv2.threshold(img_t, \
									self.game_conf.thresh, \
									self.game_conf.max_value, \
									cv2.THRESH_BINARY)
		for i in range(self.game_conf.size_pic_stack):
			self.s_t.append(img_t)
		self.s_t = np.stack((self.s_t),axis = 2) # now, prev1, prev2, prev3


	def __update_epsilon(self):
		if self.epsilon > self.game_conf.final_epsilon and self.t > self.game_conf.num_observations:
			self.epsilon -= (self.game_conf.init_epsilon - self.game_conf.final_epsilon) / self.game_conf.num_explorations

	def __store_transitions(self):
		self.deque_last = (self.s_t, self.action, self.r_t, self.s_t_1, self.terminal_b)
		self.D.append(self.deque_last)
		if len(self.D) > self.game_conf.size_replay_mem:
			self.D.popleft()

	def __update_cnn_params(self):
		if self.t > self.game_conf.num_observations:
			batch = random.sample(self.D, self.game_conf.size_batch)
			s_t_batch = []; a_t_batch = []; r_t_batch = []; s_t_1_batch = [] # can not concatenateh]

			for i in xrange(len(batch)):
				s_t_batch.append(batch[i][0]) 
				a_t_batch.append(batch[i][1])
				r_t_batch.append(batch[i][2])
				s_t_1_batch.append(batch[i][3]) # can not use [], just use append
			readout_j_1_batch = self.cnn.readout.eval(feed_dict = {self.cnn.in_layer: s_t_1_batch})[:, 0]


			r_t_1 = [] # update the Q-table value
			for i in xrange(len(batch)):
				if batch[i][4]:
					r_t_1.append(r_t_batch[i])
				else:
					r_t_1.append(r_t_batch[i] + self.game_conf.gamma * np.max(readout_j_1_batch[i]))

			self.trainer.run(feed_dict = {
				self.q_i_1: r_t_1, # future Q(s, a)
				self.action_vec: a_t_batch, # current action
				self.cnn.in_layer: s_t_batch # current state
				})

	def __print_info(self):
		if self.t % 10000 == 0:
			self.saver.save(self.sess, "saved_networks/" + self.game_conf.game_name + "/params", global_step = self.t)
		if self.t % 1000 == 0:	
			state = ""
			if self.t <= self.game_conf.num_observations:
				state = "Observe"
			elif self.t > self.game_conf.num_observations and self.t <= self.game_conf.num_observations + self.game_conf.num_explorations:
				state = "Explore"
			else:
				state = "train"
			#if self.r_t in (-1, 1):
			print(" Time: ", self.t, \
					" State: ", state, \
					" Epsilon: ", self.epsilon, \
					" Action: ", self.action_index, \
					" Reward:", self.r_t,
					" Q_max: %e"% np.max(self.readout_t))
					# "\tQ_value: ", self.readout_t[0], \
					# ", ", self.readout_t[1])
