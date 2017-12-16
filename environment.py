import gym
from ai_brain import Brain
import sys

SERIALIZE_AFTER_EPISODES = 100
REWARD_THREHSOLD = 200
CONSECUTIVE_THRESHOLD = 100

# The train parameter defines if the reward should be sent back to the brain or not
def run_using_brain(env, brain, train):
    env.render()
    current_observation = env.reset()
    while (True):
        # Get the next action from the brain
        next_action = brain.get_action_for(current_observation)
        # Apply the next_action to the environment
        current_observation, reward, done, info = env.step(next_action)
        # Re-render the screen representation
        env.render()
        # Notify the brain about the reward that was generated by it's last action
        if train:
            brain.put_reward(reward)
        # If done, the episode is finished, we should reset the environment
        if done:
         return reward


def test(env, brain):
    consecutive = 0
    while (True):
        reward = run_using_brain(env, brain, False)
        if reward > REWARD_THREHSOLD:
            consecutive+=1
        else:
            consecutive = 0
        if consecutive == 100:
            break


def train(env,brain):
    # Prepare to leave it overnight
    episodes = 0

    while(True):
        # Keep training
        if episodes % SERIALIZE_AFTER_EPISODES == 0 :
            print("Serializing model")
            brain.serialize_model()

        run_using_brain(env, brain, True)
        episodes+=1
        print("Episode {} finished!".format(episodes))

if __name__ == '__main__':
    # If we have a model passed as argument
    env = gym.make('LunarLander-v2')
    # Load the model into the brain if there was a path specified as argument
    brain = Brain(env.action_space, env.observation_space, None if len(sys.argv) == 2 else sys.argv[2])

    if sys.argv[1] == 'TRAIN':
        train(env, brain)
    else:
        test(env, brain)




