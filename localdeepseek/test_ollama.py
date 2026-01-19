"""
测试直接调用 Ollama API
"""
import requests
import json

def call_ollama_api(prompt, model="deepseek-r1:8b"):
    """
    直接调用 Ollama API
    
    Args:
        prompt: 输入提示词
        model: 模型名称，默认 deepseek-r1:8b
    
    Returns:
        模型的回复文本
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # 设置为 False 获取完整响应，True 为流式输出
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.ConnectionError:
        return "错误: 无法连接到 Ollama 服务。请确保 Ollama 正在运行。"
    except Exception as e:
        return f"错误: {str(e)}"


def call_ollama_stream(prompt, model="deepseek-r1:8b"):
    """
    流式调用 Ollama API（实时输出）
    
    Args:
        prompt: 输入提示词
        model: 模型名称
    
    Yields:
        每个响应片段
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break
    except requests.exceptions.ConnectionError:
        yield "错误: 无法连接到 Ollama 服务。请确保 Ollama 正在运行。"
    except Exception as e:
        yield f"错误: {str(e)}"


if __name__ == "__main__":
    # 测试 1: 非流式调用
    print("=" * 50)
    print("测试 1: 非流式调用")
    print("=" * 50)
    prompt = "用一句话解释什么是 Neural ODE"
    print(f"问题: {prompt}\n")
    
    response = call_ollama_api(prompt)
    print(f"回答: {response}\n")
    
    # 测试 2: 流式调用
    print("=" * 50)
    print("测试 2: 流式调用（实时输出）")
    print("=" * 50)
    prompt2 = "写一个简单的 Python 函数来计算斐波那契数列"
    print(f"问题: {prompt2}\n")
    print("回答: ", end="", flush=True)
    
    for chunk in call_ollama_stream(prompt2):
        print(chunk, end="", flush=True)
    print("\n")
