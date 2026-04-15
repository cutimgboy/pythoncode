from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    message: Annotated[list, add_messages]
    user_query: str      # 经过 LLM 理解后的用户需求总结
    search_query: str    # 优化后用于 Tavily API 的搜索查询
    search_results: str  # Tavily 搜索返回的结果
    final_answer: str    # 最终生成的答案
    step: str            # 标记当前步骤

