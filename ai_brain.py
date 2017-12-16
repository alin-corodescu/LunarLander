


class Brain:

    # Store information about observations, actions
    # TODO maybe add more elements here
    action_space = None
    observation_space = None

    # Serialize the trained model thus far, to be used later on
    def serialize_model(self):
        pass

    # Will pass the input through the network and output an action
    # This is a random action for the moment
    def get_action_for(self, current_observation):
        return self.action_space.sample()

    def __init__(self, action_space, observation_space, path = None):
        if path != None:
            self.load_model(path)
        self.action_space = action_space
        self.observation_space = observation_space


    # Stores the reward from the last action outputted by this Brain, used to train the neural network
    # As an optimization, we could batch the updates (see course slides)
    def put_reward(self, reward):
        pass

    # Loads a serialized model from the disk
    def load_model(self, path):
        pass


