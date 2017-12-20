import sys

import gym

from ai_brain import Brain

REWARD_THREHSOLD = 200
CONSECUTIVE_THRESHOLD = 100
episode = 0

# The train parameter defines if the reward should be sent back to the brain or not
def run_using_brain(env, brain, train):
    r = 0
    env.render()
    current_observation = env.reset()
    while (True):
        # Get the next action from the brain
        next_action = brain.get_action_for(current_observation, train)
        # Apply the next_action to the environment
        current_observation, reward, done, info = env.step(next_action)
        r += reward
        # Re-render the screen representation
        env.render()
        # Notify the brain about the reward that was generated by it's last action
        if train:
            brain.put_reward(reward, current_observation, done)
        # If done, the episode is finished, we should reset the environment
        if done:
            return r


def test(env, brain):
    consecutive = 0
    reward = 0
    while True:
        reward = run_using_brain(env, brain, False)
        print(reward)
        if reward > REWARD_THREHSOLD:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive == 100:
            break


def train(env, brain):
    # Prepare to leave it overnight
    episodes = 0

    # set the brain to save the model every interval seconds
    brain.set_serialization_interval(60)

    while True:
        # Keep training
        brain.serialize_model()
        reward = run_using_brain(env, brain, True)
        episodes += 1
        print(reward)
        print("Episode {} finished!".format(episodes))


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    # If we have a model passed as argument
    # Load the model into the brain if there was a path specified as argument
    brain = Brain(env.action_space, env.observation_space, None if len(sys.argv) == 2 else sys.argv[2])
    if sys.argv[1] == 'TRAIN':
        train(env, brain)
    else:
        test(env, brain)
