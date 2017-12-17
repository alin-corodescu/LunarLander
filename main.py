import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
import time

# if(np.random.choice([True, False], 1, p=[1,0])):
#     print("yes")
# env = gym.make('LunarLander-v2')
# print(env.action_space.n)
# print(env.observation_space.shape)
# model = Sequential()
# model.add(Dense(10, input_shape=(8,)))
# model.compile(optimizer='sgd', loss='categorical_crossentropy')
# observation = env.reset()
# print(type(observation))
# print(observation.reshape(1,8))
# print(observation)
# print(observation.shape)
# a = model.predict(observation.reshape(1,8))
# print(a)
# print(np.argmax(a[0]))
start = int(time.time())
print("hello")
time.sleep(5)
end = int(time.time())
print(end,start, end-start)