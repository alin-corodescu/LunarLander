import random
import time

import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import SGD


class Brain:
    # Store information about observations, actions
    action_space = None
    observation_space = None
    state = None
    action = None
    epsilon = 1
    epsilon_decay_counter = 1
    epsilon_start = 1
    epsilon_end = 0.05
    initial_random_actions = 500
    MAX_BUFFER = 5000
    BATCH_SIZE = 32
    model = None
    target_estimator = None
    discount_factor = 0.99
    buffer = []
    random_obs = 0
    start_time = int(time.time())
    serialize_file_destination = "log/model"
    serialize_time_interval = 3600
    counter = 0

    # Serialize the trained model thus far, to be used later on
    def serialize_model(self):
        current_time = int(time.time())
        if (current_time - self.start_time) >= self.serialize_time_interval * self.counter:
            self.model.save(self.serialize_file_destination + str(self.counter))
            self.counter += 1

    def set_serialization_interval(self, interval):
        if type(interval) is int and interval > 0:
            self.serialize_time_interval = interval
        else:
            raise RuntimeError("The serialization interval must be a positive number!\n")

    # Will pass the input through the network and output an action
    def get_action_for(self, current_observation, training):
        self.state = self.format_input(current_observation)
        if self.random_obs > self.initial_random_actions:
            # print("epsilon begins decaying")
            self.epsilon_decay()
        else:
            self.random_obs += 1
        if np.random.choice([True, False], 1, p=[self.epsilon, 1 - self.epsilon]) and training:
            self.action = None
            self.rand_action = self.action_space.sample()
            return self.rand_action
        else:
            # Action_vector is formatted as: [[1,2,3,4]]
            action_vector = self.model.predict(self.state)
            self.action = action_vector
            return np.argmax(action_vector[0])

    # balance between exploration and exploitation
    # At the beginning we take random actions
    # As we gain experience, we lower the posibility of taking random actions
    def epsilon_decay(self):
        if self.epsilon_decay_counter % 100 == 0:
            # print("epsilon decayed")
            self.epsilon -= 0.05

        # stop decaying after a point
        if self.epsilon_decay_counter < 2000:
            self.epsilon_decay_counter += 1

        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end


    def __init__(self, action_space, observation_space, path=None):
        self.action_space = action_space
        self.observation_space = observation_space

        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(observation_space.shape[0],)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(action_space.n))
        # self.model.add(Activation('linear'))

        optimizer = SGD(lr=0.05)
        self.model.compile(optimizer=optimizer, loss='logcosh')

        if path is not None:
            self.load_model(path)

    # Stores the reward from the last action outputted by this Brain, used to train the neural network
    # As an optimization, we could batch the updates (see course slides)

    def put_reward(self, reward, new_state, done):
        # if action is NOT random, then we train the neural network
        # if(type(self.action) == 'list'):
        #     return
        # If we need to kick out the first element
        if (len(self.buffer) == self.MAX_BUFFER):
            self.buffer.pop(0)
        self.buffer.append((self.state, self.action, reward, self.format_input(new_state), done))
        self.process_buffer()

    # Loads a serialized model from the disk
    def load_model(self, path):
        self.model = load_model(path)

    # Current_obeservation = [1,2,3,5...8]
    # After reshape
    # Current_obeservation = [[1,2,3,4,..8]]
    def format_input(self, current_observation):
        return current_observation.reshape(1, 8)

    def process_buffer(self):
        if (len(self.buffer) > self.BATCH_SIZE):
            samples = random.sample(self.buffer, self.BATCH_SIZE)
        else:
            return
        self.targets = []
        self.states = []
        self.tar_arg = 0
        for (s, a, r, s_prim, done) in samples:
            if a is not None:
                self.tar_arg = np.argmax(a[0])
            else:
                self.tar_arg = self.rand_action
            q = self.model.predict(s)
            target = q
            q_prim = self.model.predict(s_prim)
            # np.argmax(a[0]) = the action: s -> s_prim
            # q_prim[0][0] = estimated reward starting from s_prim and apllying action 0
            # q_prim = [0][1] = estimated reward starting from s_prim and apllying action 1
            target[0][self.tar_arg] = r + self.discount_factor * max(q_prim[0]) if not done else 0
            self.targets.append(target[0])
            self.states.append(s[0])
        # aici se produce eroare, uitate si la __init__ cum am initalizat vectorii, posibil/PROBABIL am gresit pe acolo.
        self.targets = np.array(self.targets)
        self.states = np.array(self.states)
        self.model.fit(self.states, self.targets)
