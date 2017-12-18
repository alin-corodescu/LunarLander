import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import SGD
import time


class Brain:
    # Store information about observations, actions
    action_space = None
    observation_space = None
    state = None
    action = 0
    epsilon = 0.1
    MAX_BUFFER = 10000
    model = None
    discount_factor = 0.8
    buffer = []
    targets = np.array([])
    states = np.array([])
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
    # This is a random action for the moment
    def get_action_for(self, current_observation):
        self.state = self.format_input(current_observation)
        if np.random.choice([True, False], 1, p=[self.epsilon, 1 - self.epsilon]):
            return self.action_space.sample()
        else:
            # Action_vector is formatted as: [[1,2,3,4]]
            action_vector = self.model.predict(self.state)
            self.action = action_vector
            return np.argmax(action_vector[0])

    def __init__(self, action_space, observation_space, path=None):
        self.action_space = action_space
        self.observation_space = observation_space

        self.model = Sequential()
        self.model.add(Dense(100, input_shape=(observation_space.shape[0],)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(action_space.n))
        self.model.add(Activation('linear'))

        optimizer = SGD(lr=0.2, momentum=0.0)
        self.model.compile(optimizer=optimizer, loss='logcosh')

        if path is not None:
            self.load_model(path)

    # Stores the reward from the last action outputted by this Brain, used to train the neural network
    # As an optimization, we could batch the updates (see course slides)

    def put_reward(self, reward, new_state):
        # if action is NOT random, then we train the neural network
        # if(type(self.action) == 'list'):
        #     return
        self.buffer.append((self.state, self.action, reward, self.format_input(new_state)))

        if len(self.buffer) == self.MAX_BUFFER:
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
        np.random.shuffle(self.buffer)
        for (s, a, r, s_prim) in self.buffer[:len(self.buffer)]:
            q = self.model.predict(s)
            target = q
            q_prim = self.model.predict(s_prim)
            # np.argmax(a[0]) = the action: s -> s_prim
            # q_prim[0][0] = estimated reward starting from s_prim and apllying action 0
            # q_prim = [0][1] = estimated reward starting from s_prim and apllying action 1
            target[0][np.argmax(a[0])] = r + self.discount_factor * max(q_prim[0])
            np.append(self.targets,target)
            np.append(self.states,s)
        # aici se produce eroare, uitate si la __init__ cum am initalizat vectorii, posibil/PROBABIL am gresit pe acolo.
        self.model.fit(self.states, self.targets, batch_size=32)
        self.buffer.clear()
