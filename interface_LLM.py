# 从同级目录的llm模块中导入处理远程API和本地LLM的接口类
from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM

# 定义一个统一的LLM接口类，用于封装本地和远程LLM的调用逻辑
class InterfaceLLM:
    # 初始化方法，接收API端点、密钥、模型名称、是否使用本地LLM、本地LLM地址和调试模式等参数
    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode):
        # 将传入的参数赋值给实例变量，用于后续使用
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url

        # 打印提示信息，指示正在检查LLM API连接
        print("- check LLM API")

        # 如果设置为使用本地LLM
        if self.llm_use_local:
            # 打印提示信息，说明正在使用本地LLM部署
            print('local llm delopyment is used ...')
            
            # 检查本地LLM的URL是否为空或默认值，如果是则打印错误信息并退出程序
            if self.llm_local_url == None or self.llm_local_url == 'xxx' :
                print(">> Stop with empty url for local llm !")
                exit()

            # 初始化本地LLM接口实例，传入本地LLM的URL
            self.interface_llm = InterfaceLocalLLM(
                self.llm_local_url
            )

        # 如果不使用本地LLM，则使用远程LLM API
        else:
            # 打印提示信息，说明正在使用远程LLM API
            print('remote llm api is used ...')

            # 检查远程API的端点和密钥是否为空或默认值，如果是则打印错误信息并退出程序
            if self.api_key == None or self.api_endpoint ==None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
                print(">> Stop with wrong API setting: Set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !")
                exit()

            # 初始化远程API接口实例，传入API端点、密钥、模型名称和调试模式
            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
            )

        # 发送一个简单的"1+1=?"请求来测试LLM连接是否正常
        res = self.interface_llm.get_response("1+1=?")

        # 如果测试响应为空，说明LLM连接存在问题，打印错误信息并退出程序
        if res == None:
            print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
            exit()

    # 定义获取响应的方法，接收提示词内容并返回LLM的响应结果
    def get_response(self, prompt_content):
        # 调用内部封装的LLM接口（本地或远程）的get_response方法获取响应
        response = self.interface_llm.get_response(prompt_content)

        # 返回获取到的响应
        return response
