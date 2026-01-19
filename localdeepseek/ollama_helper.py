"""
Ollama API 辅助函数
简化调用 deepseek-r1:8b 模型
"""
import requests
import json
from typing import Optional, Generator


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:8b"):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama 服务地址
            model: 模型名称
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, stream: bool = False) -> Optional[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            stream: 是否使用流式输出
        
        Returns:
            生成的文本（非流式）或 None（流式）
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            if stream:
                # 流式输出
                response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
                response.raise_for_status()
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            chunk = data["response"]
                            full_response += chunk
                            yield chunk
                        if data.get("done", False):
                            break
                return None
            else:
                # 非流式输出
                response = requests.post(self.api_url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
        except requests.exceptions.ConnectionError:
            error_msg = "错误: 无法连接到 Ollama 服务。请确保 Ollama 正在运行。"
            if stream:
                yield error_msg
                return None
            return error_msg
        except Exception as e:
            error_msg = f"错误: {str(e)}"
            if stream:
                yield error_msg
                return None
            return error_msg
    
    def chat(self, messages: list, stream: bool = False) -> Optional[str]:
        """
        对话模式（如果模型支持）
        
        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            stream: 是否使用流式输出
        
        Returns:
            生成的文本
        """
        # 将消息列表转换为单个 prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate(prompt, stream=stream)


# 便捷函数
def quick_ask(question: str, model: str = "deepseek-r1:8b") -> str:
    """
    快速提问（非流式）
    
    Args:
        question: 问题
        model: 模型名称
    
    Returns:
        回答
    """
    client = OllamaClient(model=model)
    return client.generate(question, stream=False)


def quick_ask_stream(question: str, model: str = "deepseek-r1:8b") -> Generator:
    """
    快速提问（流式输出）
    
    Args:
        question: 问题
        model: 模型名称
    
    Yields:
        回答片段
    """
    client = OllamaClient(model=model)
    for chunk in client.generate(question, stream=True):
        yield chunk


if __name__ == "__main__":
    # 示例 1: 简单调用
    print("示例 1: 简单调用")
    print("-" * 50)
    answer = quick_ask("什么是 Neural ODE？用一句话回答。")
    print(f"回答: {answer}\n")
    
    # 示例 2: 使用客户端类
    print("示例 2: 使用客户端类")
    print("-" * 50)
    client = OllamaClient()
    answer2 = client.generate("解释一下 Lyapunov 稳定性理论", stream=False)
    print(f"回答: {answer2}\n")
    
    # 示例 3: 流式输出
    print("示例 3: 流式输出")
    print("-" * 50)
    print("回答: ", end="", flush=True)
    for chunk in quick_ask_stream("写一个简单的 Python 函数计算平方"):
        print(chunk, end="", flush=True)
    print("\n")
