import openai
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="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",
    # 如果是自定义端点（如使用代理或国内服务）
    base_url="https://openrouter.ai/api/v1"
)

def basic_chat():
    """基础对话示例"""
    try:
        response = client.chat.completions.create(
            model="inclusionai/ling-2.6-flash:free",  # 或 "gpt-4"
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": "介绍一下Python的优势"}
            ],
            temperature=0.7,  # 控制随机性 0-2
            max_tokens=500,   # 最大生成token数
            top_p=1.0,        # 核采样参数
            frequency_penalty=0.0,  # 频率惩罚
            presence_penalty=0.0    # 存在惩罚
        )
        
        # 获取回复内容
        message = response.choices[0].message.content
        print(f"回复: {message}")
        
        # 查看使用情况
        print(f"Token使用: {response.usage}")
        
        return message
        
    except openai.APIError as e:
        print(f"API错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")

# 调用示例
basic_chat()