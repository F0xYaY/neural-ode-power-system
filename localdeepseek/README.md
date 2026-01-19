# Local DeepSeek R1 with Ollama

本地运行 DeepSeek R1 模型的工具集，使用 Ollama 作为推理引擎。

## 文件说明

- **`ollama_helper.py`**: Ollama API 客户端封装，提供便捷的调用接口
- **`test_ollama.py`**: 基础测试脚本，演示如何调用 Ollama API

## 使用方法

### 1. 确保 Ollama 已安装并运行

```bash
# 检查 Ollama 是否运行
ollama list

# 如果没有 deepseek-r1:8b 模型，先下载
ollama pull deepseek-r1:8b
```

### 2. 安装依赖

```bash
pip install requests
```

### 3. 使用示例

#### 简单调用

```python
from ollama_helper import quick_ask

answer = quick_ask("什么是 Neural ODE？")
print(answer)
```

#### 使用客户端类

```python
from ollama_helper import OllamaClient

client = OllamaClient(model="deepseek-r1:8b")
answer = client.generate("解释一下 Lyapunov 稳定性", stream=False)
print(answer)
```

#### 流式输出

```python
from ollama_helper import quick_ask_stream

for chunk in quick_ask_stream("写一个 Python 函数"):
    print(chunk, end="", flush=True)
```

## API 说明

### OllamaClient 类

- `generate(prompt, stream=False)`: 生成文本
- `chat(messages, stream=False)`: 对话模式

### 便捷函数

- `quick_ask(question, model="deepseek-r1:8b")`: 快速提问（非流式）
- `quick_ask_stream(question, model="deepseek-r1:8b")`: 快速提问（流式）

## 模型信息

- **模型**: deepseek-r1:8b
- **大小**: 约 5.2 GB
- **参数**: 8.2B
- **上下文长度**: 131072 tokens
- **量化**: Q4_K_M

## 注意事项

- 确保 Ollama 服务正在运行（默认端口 11434）
- 首次使用需要下载模型，可能需要一些时间
- 流式输出适合实时显示，非流式适合批量处理
