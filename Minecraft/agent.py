import gym
import minerl
import matplotlib.pyplot as plt


# Create environment, this should use previously created environment
env = gym.make('MineRLNavigateDense-v0')

obs = env.reset()



done = False

total = 0
totalList = []
step = 0
stepList = []

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total += reward
    step += 1
    totalList.append(total)
    stepList.append(step)

plt.plot(stepList, totalList)
plt.xlabel('Steps')
plt.ylabel('Return')

plt.title('Total return per number of steps - Random agent')

plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
