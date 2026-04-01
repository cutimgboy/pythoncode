from tools import search, ToolExecutor
from template import REACT_PROMPT_TEMPLATE
from agent import ReActAgent
from client import HelloAgentsLLM

# 加载客户端
llmClient = HelloAgentsLLM()

# 加载工具
toolExecutor = ToolExecutor()

toolExecutor.registerTool("search", "检索工具", search)

# 运行 agent
agent = ReActAgent(llmClient, toolExecutor)

agent.run("华为手机的最新型号")
