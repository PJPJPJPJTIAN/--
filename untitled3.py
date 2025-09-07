# 导入必要的模块：re用于正则表达式处理，time用于时间相关操作，InterfaceLLM用于与语言模型交互
import re
import time
from ...llm.interface_LLM import InterfaceLLM

# 定义Evolution类，用于实现进化算法逻辑，通过调用语言模型生成和优化算法代码
class Evolution():

    # 初始化方法，设置语言模型参数、提示词配置等
    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # 设置提示词相关属性，从prompts对象中获取任务描述、函数名、输入输出等信息
        self.prompt_task         = prompts.get_task()  # 任务描述
        self.prompt_func_name    = prompts.get_func_name()  # 目标函数名
        self.prompt_func_inputs  = prompts.get_func_inputs()  # 函数输入参数列表
        self.prompt_func_outputs = prompts.get_func_outputs()  # 函数输出参数列表
        self.prompt_inout_inf    = prompts.get_inout_inf()  # 输入输出详细信息
        self.prompt_other_inf    = prompts.get_other_inf()  # 其他补充信息

        # 处理输入参数列表，转换为字符串格式（如"'a', 'b'"）
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        # 处理输出参数列表，转换为字符串格式
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # 设置语言模型相关参数
        self.api_endpoint = api_endpoint  # API端点
        self.api_key = api_key  # API密钥
        self.model_LLM = model_LLM  # 语言模型名称
        self.debug_mode = debug_mode  # 调试模式开关（控制是否打印提示信息）

        # 初始化语言模型接口，支持本地或远程模型
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    # 生成初始算法的提示词（i1操作：随机初始化个体）
    def get_prompt_i1(self):
        
        # 构造提示词内容：要求模型先描述算法（用大括号包裹），再实现指定函数，包含输入输出要求
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

        
    # 生成交叉操作e1的提示词（基于现有算法生成全新算法）
    def get_prompt_e1(self,indivs):
        # 拼接现有算法的描述和代码作为提示词的一部分
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        # 构造提示词：要求基于给定算法生成形式完全不同的新算法
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    # 生成交叉操作e2的提示词（基于现有算法的共同思想生成新算法）
    def get_prompt_e2(self,indivs):
        # 拼接现有算法的描述和代码
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        # 构造提示词：要求先识别现有算法的共同核心思想，再基于该思想生成新算法
        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    # 生成变异操作m1的提示词（对现有算法进行形式修改）
    def get_prompt_m1(self,indiv1):
        # 构造提示词：要求基于输入算法生成形式不同的修改版本
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    # 生成变异操作m2的提示词（修改现有算法的参数设置）
    def get_prompt_m2(self,indiv1):
        # 构造提示词：要求识别算法的主要参数，并生成参数设置不同的新版本
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content
    
    # 生成变异操作m3的提示词（简化算法以增强泛化能力）
    def get_prompt_m3(self,indiv1):
        # 构造提示词：要求识别算法组件，分析过拟合风险，简化组件以提升泛化能力
        prompt_content = "First, you need to identify the main components in the function below. \
Next, analyze whether any of these components can be overfit to the in-distribution instances. \
Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. \
Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"+indiv1['code']+"\n"\
+self.prompt_inout_inf+"\n"+"Do not give additional explanations."
        return prompt_content


    # 内部方法：调用语言模型获取算法代码和描述，并提取关键信息
    def _get_alg(self,prompt_content):

        # 调用语言模型接口获取响应
        response = self.interface_llm.get_response(prompt_content)

        # 从响应中提取算法描述（优先匹配大括号内的内容，否则根据关键词截取）
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

        # 从响应中提取代码（优先匹配import到return的内容，否则匹配def到return）
        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        # 若提取失败，重试最多3次
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            # 重新提取算法描述和代码
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
                
            if n_retry > 3:
                break
            n_retry +=1

        # 提取到的结果取第一个元素，并拼接完整代码（补充返回值）
        algorithm = algorithm[0]
        code = code[0] 
        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 

        # 返回完整代码和算法描述
        return [code_all, algorithm]


    # 初始算法生成（i1操作）：调用提示词生成方法，获取并返回算法代码和描述
    def i1(self):

        prompt_content = self.get_prompt_i1()

        # 调试模式下打印提示词并等待用户确认
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        # 获取算法代码和描述
        [code_all, algorithm] = self._get_alg(prompt_content)

        # 调试模式下打印结果并等待用户确认
        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    # 交叉操作e1：基于父代算法生成新算法
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    # 交叉操作e2：基于父代算法的共同思想生成新算法
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    # 变异操作m1：修改算法形式生成新算法
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    # 变异操作m2：修改算法参数生成新算法
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    # 变异操作m3：简化算法以增强泛化能力
    def m3(self,parents):
      
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m3 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]