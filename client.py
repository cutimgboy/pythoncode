import requests

class OpenAICompatibleClient:
    """
    纯 requests 实现，兼容你提供的新模型接口 https://api.edgefn.net/v1/chat/completions
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        # 自动拼接完整接口地址
        self.api_url = base_url.rstrip("/") + "/chat/completions"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")

        # 请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 消息格式（兼容 system + user）
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 请求体
        data = {
            "model": self.model,
            "messages": messages
        }

        try:
            # 发送 POST 请求
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # 自动抛出 HTTP 错误

            result = response.json()
            # 标准 OpenAI 格式取值
            answer = result["choices"][0]["message"]["content"]

            print("大语言模型响应成功。")
            return answer

        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误: 调用语言模型服务时出错。"

if __name__ == "__main__":
    client = OpenAICompatibleClient(
        model="DeepSeek-R1-0528-Qwen3-8B",
        api_key="sk-ZxiqmnoVAC3rwiWJ1289Fa38C32b4473A70e5eF238A7B582",
        base_url="https://api.edgefn.net/v1"
    )

    # 调用
    response = client.generate(
        prompt="Hello, how are you?",
        system_prompt="你是一个有用的助手"
    )

    print(response)