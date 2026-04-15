from agent import PlanAndSolveAgent
from client import HelloAgentsLLM

# 加载客户端
llmClient = HelloAgentsLLM()

# agent
agent = PlanAndSolveAgent(llmClient)

agent.run("华为手机的最新型号")
