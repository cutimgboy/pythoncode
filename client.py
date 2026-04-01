import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    纯 requests 实现，兼容你提供的新模型接口 https://api.edgefn.net/v1/chat/completions
    """
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        # 自动拼接完整接口地址
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_url = self.base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """调用LLM API流式输出并返回完整结果"""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            print("✅ 大语言模型响应成功:")
            collected_content = []

            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8").strip()
                if not line_text.startswith("data: "):
                    continue

                json_str = line_text[6:]  # 去掉 "data: "

                # 结束标志
                if json_str.strip() == "[DONE]":
                    break

                try:
                    result = json.loads(json_str)
                except:
                    continue

                # ✅ 超级安全取值，彻底解决 list index out of range
                choices = result.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)

            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return f"[错误] {str(e)}"

if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)