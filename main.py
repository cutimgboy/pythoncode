from agent import ReflectionAgent
from client import HelloAgentsLLM

# 加载客户端
llmClient = HelloAgentsLLM()

# agent
agent = ReflectionAgent(llmClient)

agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")
