# 导入用于HTTP通信的http.client模块和用于JSON数据处理的json模块
import http.client
import json

# 定义一个用于与远程LLM API交互的接口类
class InterfaceAPI:
    # 初始化方法，接收API端点、API密钥、模型名称和调试模式参数
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        # 将传入的API端点赋值给实例变量，用于后续连接
        self.api_endpoint = api_endpoint
        # 将传入的API密钥赋值给实例变量，用于身份验证
        self.api_key = api_key
        # 将传入的模型名称赋值给实例变量，指定调用的LLM模型
        self.model_LLM = model_LLM
        # 将传入的调试模式标志赋值给实例变量，控制调试信息输出
        self.debug_mode = debug_mode
        # 设置最大重试次数为5次，用于API调用失败时的重试机制
        self.n_trial = 5

    # 定义获取LLM响应的方法，接收提示词内容作为参数
    def get_response(self, prompt_content):
        # 构造API请求的JSON payload，包含模型名称和对话消息（仅用户提示词）
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,  # 指定使用的模型
                "messages": [
                    # 注释掉的系统角色消息，原本用于定义助手行为
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}  # 用户输入的提示词
                ],
            }
        )

        # 构造HTTP请求头，包含身份验证、用户代理、内容类型等信息
        headers = {
            "Authorization": "Bearer " + self.api_key,  # 使用Bearer令牌进行身份验证
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",  # 模拟Apifox的用户代理
            "Content-Type": "application/json",  # 声明请求体为JSON格式
            "x-api2d-no-cache": 1,  # 禁用缓存，确保获取最新响应
        }
        
        # 初始化响应变量为None
        response = None
        # 初始化重试计数器为1
        n_trial = 1
        # 进入循环，直到成功获取响应或达到最大重试次数
        while True:
            # 重试计数器加1
            n_trial += 1
            # 如果重试次数超过最大限制，返回当前响应（可能为None）
            if n_trial > self.n_trial:
                return response
            try:
                # 建立与API端点的HTTPS连接
                conn = http.client.HTTPSConnection(self.api_endpoint)
                # 发送POST请求到指定的API路径（聊天补全接口），携带payload和 headers
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                # 获取服务器的响应
                res = conn.getresponse()
                # 读取响应数据
                data = res.read()
                # 将响应数据解析为JSON格式
                json_data = json.loads(data)
                # 提取响应中的生成内容（取第一个选择的消息内容）
                response = json_data["choices"][0]["message"]["content"]
                # 成功获取响应后跳出循环
                break
            except:
                # 如果处于调试模式，打印API调用错误信息
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                # 发生异常时继续循环重试
                continue
        
        # 返回从API获取的响应内容
        return response