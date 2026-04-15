from stateManager import SearchState

def underStand_query_node(state: SearchState) -> dict:
    """
    步骤 1：理解用户查询并生成搜索关键词
    """
    user_message = state["message"][-1].content
    understand_prompt = f"""
        分析用户的查询："{user_message}"
        请完成两个任务：
        1. 简洁总结用户想要了解什么
        2. 生成最适合搜索引擎的关键词(中英文均可，要精确)
        
        格式：
        理解：[用户需求总结]
        搜索词：[最佳搜索关键词]
    """
    response = llm.invoke([SystemMessage(content=understand_prompt)])

