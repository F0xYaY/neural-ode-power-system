"""
简单的交互式聊天脚本
"""
from ollama_helper import OllamaClient

def main():
    print("=" * 60)
    print("DeepSeek R1 8B 聊天助手")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出\n")
    
    client = OllamaClient(model="deepseek-r1:8b")
    
    while True:
        try:
            question = input("你: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n再见！")
                break
            
            if not question:
                continue
            
            print("\nDeepSeek: ", end="", flush=True)
            # 使用流式输出，实时显示
            for chunk in client.generate(question, stream=True):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")

if __name__ == "__main__":
    main()
