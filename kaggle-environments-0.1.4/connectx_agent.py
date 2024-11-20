from kaggle_environments import evaluate, make, utils
from random import choice

# 创建 ConnectX 环境
env = make("connectx", debug=True)
env.render()

# 定义智能体
def my_agent(observation, configuration):
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

# 测试智能体
env.reset()
env.run([my_agent, "random"])
env.render(mode="ipython", width=500, height=450)

# 训练智能体
trainer = env.train([None, "random"])
observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
env.render()

# 评估智能体
def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
