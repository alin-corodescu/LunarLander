import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import SGD


class Brain:

    # Store information about observations, actions
    # TODO maybe add more elements here
    action_space = None
    observation_space = None
    state = None
    action = 0
    epsilon = 0.1
    MAX_BUFFER = 25
    model = None
    discount_factor = 0.8
    buffer = []
    # Serialize the trained model thus far, to be used later on
    def serialize_model(self):
        # TODO add a diferent
        self.model.save("model")

    # Will pass the input through the network and output an action
    # This is a random action for the moment
    def get_action_for(self, current_observation):
        self.state = self.format_input(current_observation)
        if np.random.choice([True, False], 1, p=[self.epsilon, 1-self.epsilon]):
            return self.action_space.sample()
        else:
            action_vector = self.model.predict(self.state)
            self.action = action_vector
            return np.argmax(action_vector[0])

    def __init__(self, action_space, observation_space, path = None):
        if path != None:
            self.load_model(path)
        self.action_space = action_space
        self.observation_space = observation_space

        self.model = Sequential()
        self.model.add(Dense(100, input_shape=(observation_space.shape[0],)))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(action_space.n))
        self.model.add(Activation('sigmoid'))

        optimizer = SGD(lr=0.2, momentum=0.0)
        self.model.compile(optimizer=optimizer, loss='mse')

    # Stores the reward from the last action outputted by this Brain, used to train the neural network
    # As an optimization, we could batch the updates (see course slides)
    def put_reward(self, reward, new_state):
        self.buffer.append((self.state, self.action, reward, self.format_input(new_state)))

        if (len(self.buffer) == self.MAX_BUFFER):
            self.clear_buffer()




    # Loads a serialized model from the disk
    def load_model(self, path):
        model = load_model(path)

    def format_input(self, current_observation):
        return current_observation.reshape(1,8)

    def clear_buffer(self):
        np.random.shuffle(self.buffer)
        #         Sample from the result
        for (s, a, r, s_prim) in self.buffer[:len(self.buffer)]:
            q = self.model.predict(s)
            target = q
            q_prim = self.model.predict(s_prim)
            target[0][np.argmax(a[0])] = r + self.discount_factor * max(q_prim[0])
            self.model.fit(s, target)
        self.buffer.clear()


