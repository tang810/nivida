import re
import os
import ast
import sys
import json
import time
import runpy
import chardet
import mimetypes
import traceback
import importlib
import numpy as np
import pandas as pd
from typing import ClassVar

from tornado.process import task_id

from alpha.team import Team
from alpha.roles import Role
from alpha.logs import logger
from alpha.schema import Message
from alpha.actions import Action, UserRequirement

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


from src.llm_utils import SeLLM
from src.llm_utils import load_config
from src.Tools import EDA_Tools
from src.Tools.date_utils import check_date_column
from src.Tools.plot_utils import boxplot,heatmap_plot,violin_plot,density_plot
from src.oss_utils import download_to_file,oss_upload_by_path,get_image_url

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

from dotenv import load_dotenv
from utils import read_data_file, read_data_file, format_preview_for_llm

load_dotenv()
# 读取环境变量
server_base = os.getenv('server_base')
base_path = os.getenv('base_path')

#IMG_URL = os.getenv('IMG_URL')

handler = {"sink": sys.stdout, "level": "ERROR"}
logger.configure(handlers=[handler])
init_file_names=None
df_data_list = None


async def analyze_failed_file_with_llm(websocket, llm, file_preview, error_msg, instruction):
    """
    让LLM分析失败的文件并提供建议
    """
    
    # 构建给LLM的prompt
    FAILED_FILE_ANALYSIS_PROMPT = """
    你是数据分析专家。用户上传了一个文件，但文件读取失败了。
    不过我获取了一些文件的基本信息，请你基于这些信息进行分析和建议。

    用户的问题是：{instruction}
    
    文件错误信息：{error_msg}
    
    文件基本信息：
    {preview_info}
    
    请你：
    1. 分析文件内容和结构
    2. 解释为什么文件读取失败
    3. 提供具体的修复建议
    4. 如果可能，推测用户想要进行什么样的分析
    
    请给出详细且有用的分析和建议：
    """
    
    # 格式化文件预览信息
    preview_text = format_preview_for_llm(file_preview)
    
    prompt = FAILED_FILE_ANALYSIS_PROMPT.format(
        instruction=instruction,
        error_msg=error_msg,
        preview_info=preview_text
    )
    
    # 调用LLM
    message = [llm._default_system_msg()]
    message.append(llm._user_msg(prompt))
    response = await llm.acompletion_text(message, temperature=0.7, timeout=3)
    
    await websocket.send_text("## 文件分析和建议\n\n")
    
    # 流式输出LLM的回答
    async for chunk in response:
        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
        await websocket.send_text(chunk_message)
    
    await websocket.send_text("\n\n---\n\n")


class Input_Analysis(Action):
    name: str = "基本信息解读"
    desc: str = "数据文件基本信息解读"
    
    data: ClassVar[str] = None
    
    PROMPT_TEMPLATE: str  = """
    你是统计分析专家。你需要根据输入数据的python统计量对输入数据进行初步描述、解释以及分析：
    
    数据的基本信息:
        前五行：
        {head}
        维度信息：
        {shape}
        统计描述：
        {desc}
        结构信息：
        {info}
        
    用户的问题是：{instruction}
    
    请给出你的详细分析,结果仅以表格形式输出：
    """

    PROMPT_TEMPLATE_EXTRACT_DATE: str  = """
    你需要根据用户输入的{instruction}，并结合数据的基本信息，提取日期列名或索引，并返回日期列名或索引。或者其他如"index"等可以作为时间序列索引的列名。
    如果用户没有指定日期列名或者索引，并且你没有找到日期列名或索引，直接返回空值。

    数据的基本信息：
        前五行：
        {head}
        维度信息：
        {shape}
        统计描述：
        {desc}
        结构信息：
        {info}

    请注意，无需其他描述性文字，直接给出列名或者索引或者空结果，加上中括号[]，即返回值是[列名]、[索引]、[]这三者之一，不需要使用任何单引号或者双引号，如[date]、[3]、[]等。
    如果用户指定第几列是日期列，你需要直接返回该列的位置，而不是列名，即[列索引]。
    如果日期、时间等列与年份、月份等同时存在，优先选择名为日期、时间的列作为时间戳。
    """

    PROMPT_TEMPLATE_GENERAL: str  = """
    你是统计分析专家。你需要根据用户的问题来做基本的数据科学与统计分析相关领域概念解读。
    用户的问题是：{instruction}
    
    请给出你的详细分析,一步步的专业解读：
    """
    
    #2、结合统计分析工具{tool}的原理，对上面结论进行系统性总结以及提出批判性的思考
    PROMPT_TEMPLATE_STATISTIC: str  = """
    你是数据科学家，擅长统计分析，你需要按照以下步骤一步步分析，逻辑清晰，不要输出任何代码，请使用中文分析：
    1、结合数据分析结果{analysis},对结果{result}进行专业的综合性分析与解读 
        
    请给出你的回答：
    """

    async def run(self, instruction: str, *args):
        websocket = args[0]
        user_name,taskid, file_names = args[1], args[2], args[3]

        # 加载配置
        config = load_config("config/config.yaml")
        # 创建LLM
        llm = SeLLM(
                base_url=config["base_url_1"],
                api_key=config["api_key"]
        )
        # 创建LLM
        llm_2 = SeLLM(
                base_url=config["base_url_2"],
                api_key=config["api_key"]
        )
        
        global df_data_list
        global init_file_names
        global date_name_list
        df_data_list =[]
        init_file_names=file_names
        date_name_list =[]
        
        save_path="upload/{}/{}".format(user_name,taskid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 通过云存储把指定文件下载到本地
        download_to_file("science-backend" ,"upload/{}".format(user_name),save_path,file_names)


        print('file_names:{}'.format(file_names))
        if len(file_names) ==0:
            await websocket.send_text("没有找到文件呢~")
        else:
            for file_name in file_names:
                file_path = os.path.join("upload", user_name, taskid, file_name)
     
                if not os.path.exists(file_path):
                    await websocket.send_text("没有找到文件{}呢~".format(file_name))
                    print("文件不存在！请检查是否上传成功！")
                    continue  # 继续处理下一个文件，而不是关闭连接
                
                # 修改这里：获取三个返回值
                code, result_data, file_preview = await read_data_file(file_path, websocket)
                
                if code == -1:
                    # 文件读取失败，但仍要让LLM处理
                    await websocket.send_text("文件读取失败，但让我看看能分析出什么...\n\n")
                    
                    # 发送文件预览信息
                    preview_text = format_preview_for_llm(file_preview)
                    await websocket.send_text(f"```\n{preview_text}\n```\n\n")
                    
                    # 让LLM基于预览信息进行分析
                    await analyze_failed_file_with_llm(websocket, llm, file_preview, result_data, instruction)
                    
                    print(f"文件处理失败但已分析: {result_data}")
                    continue  # 继续处理下一个文件
                
                elif code == 0:
                    # 文件读取成功，正常处理
                    data = result_data
                    df_head = data.head()
                    df_tail = data.tail()
                    df_shape = data.shape
                    df_desc = data.describe(include="all")
                    df_info = data.info()

                    # 使用prompt获取日期列名
                    prompt = self.PROMPT_TEMPLATE_EXTRACT_DATE.format(instruction=instruction, head=df_head, 
                            tail=df_tail, shape=df_shape, desc=df_desc, info=df_info)
                    # 接受llm返回值并异步返回
                    collected_messages = []
                    message = [self.llm._default_system_msg()]
                    message.append(self.llm._user_msg(prompt))
                    response = await llm.acompletion_text(message,temperature=0.7,timeout=3)
                    async for chunk in response:
                        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
                        # await websocket.send_text(chunk_message)
                        collected_messages.append(chunk_message)
                    full_reply_content = "".join(collected_messages)
                    # 检查日期列
                    success, col_name, error_msg = check_date_column(full_reply_content[1:-1], data)
                    date_name_list.append(col_name)
                    if success:
                        await websocket.send_text("")
                    else:
                        # 返回为空，用户指定的列不存在，或者没有指定日期列。新建一列index作为样本编号
                        await websocket.send_text(error_msg)
                    df_data_list.append(data)

                else:
                    print("异常code:{}".format(str(code)))
                    await websocket.close()
        
        # 遍历分析每个数据集
        all_full_reply_content=""
        for idx,data in enumerate(df_data_list):
            print("使用 '{}' 作为日期列，".format(date_name_list[idx]))
            await websocket.send_text("分析{}数据集：\n\n".format(file_names[idx] if len(file_names)>0 else "默认"))
            # data.head()中文对齐
            pd.set_option('display.unicode.ambiguous_as_wide', True)
            pd.set_option('display.unicode.east_asian_width', True)
            await websocket.send_text("```bash\n{}\n```".format(data.head()))
            await websocket.send_text("\n\n\n")
            
            # 获取当前时间，并格式化为字符串
            current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

            img_save_path = 'images/{}'.format(taskid)
            if not os.path.exists(img_save_path):
                os.mkdir(img_save_path)

            prompt = self.PROMPT_TEMPLATE.format(instruction=instruction, head=df_head, tail=df_tail, 
                    shape=df_shape,desc=df_desc, info=df_info)
            message = [self.llm._default_system_msg()]
            message.append(self.llm._user_msg(prompt))
            response = await llm.acompletion_text(message,temperature=0.7,timeout=3)
            collected_messages=[]
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
                #chunk_message = chunk_message.replace('<think>', '(。≖ˇェˇ≖｡) 思考中...')
                #chunk_message = chunk_message.replace('</think>', '<(▰˘◡˘▰)> 好啦')
                await websocket.send_text(chunk_message)
            collected_messages.append(chunk_message)
            await websocket.send_text("\n\n")
            analysis = ''.join(collected_messages)

            # density_plot
            density_name='density_plot_{}_{}.png'.format(file_names[idx] if len(file_names)>0 else str(idx),current_time)
            density_path = os.path.join(img_save_path,density_name )
            density_plot(data,density_path)
            # 上传到云存储
            oss_upload_by_path("science-images" ,"{}/{}/{}".format(user_name,taskid,density_name),density_path)
            density_image_url=get_image_url("science-images","{}/{}/{}".format(user_name,taskid,density_name))
            # 拼接markdown图片地址
            density_path_img_str = "\r\r![img]({})".format(density_image_url)
            await websocket.send_text("\n\n绘制所有数值列的密度图:\n" if len(file_names)==0 else "\n\n绘制数据集:{}密度图:\n".format(file_names[idx]))
            await websocket.send_text("\n\n\n")
            await websocket.send_text(density_path_img_str)

            '''
            # heatmap
            heatmap_name='heatmap_{}_{}.png'.format(file_names[idx] if len(file_names)>0 else str(idx),current_time)
            heatmap_path = os.path.join(img_save_path,heatmap_name )
            heatmap_plot(data,heatmap_path)
            # 上传到云存储
            oss_upload_by_path("science-images" ,"{}/{}/{}".format(user_name,taskid,heatmap_name),heatmap_path)
            heatmap_name_image_url=get_image_url("science-images","{}/{}/{}".format(user_name,taskid,heatmap_name))
            # 拼接markdown图片地址
            heatmap_img_str = "\r\r![img]({})".format(heatmap_name_image_url)
            await websocket.send_text("\n\n绘制所有数值列的热力图:\n" if len(file_names)==0 else "\n\n绘制数据集:{}热力图:\n".format(file_names[idx]) )
            #await websocket.send_text("\n\n\n")
            await websocket.send_text(heatmap_img_str)

            # boxplot
            box_plot_name='boxplot_{}_{}.png'.format(file_names[idx] if len(file_names)>0 else str(idx),current_time)
            boxplot_path = os.path.join(img_save_path,box_plot_name )
            boxplot(data,boxplot_path)
            # 上传到云存储
            oss_upload_by_path("science-images" ,"{}/{}/{}".format(user_name,taskid,box_plot_name),boxplot_path)
            boxplot_image_url=get_image_url("science-images","{}/{}/{}".format(user_name,taskid,box_plot_name))
            # 拼接markdown图片地址
            boxplot_img_str = "\r\r![img]({})".format(boxplot_image_url)
            await websocket.send_text("\n\n绘制所有数值列的箱形图:\n" if len(file_names)==0 else "\n\n绘制数据 {} 箱形图:\n".format(file_names[idx]))
            #await websocket.send_text("\n\n\n")
            await websocket.send_text(boxplot_img_str)
            '''

            function_instruction = EDA_Tools.__doc__
            tool_lst = ["acf_and_pacf","total_trend", "fft_periodic", "CBLOF"]
            #tool_lst = ["CBLOF", "fft_periodic", "acf_and_pacf"]
            #tool_lst = ["acf_and_pacf","z_score","CBLOF", "fft_periodic"]
            for tool_name in tool_lst:
                results = []
                img_results = ""
                full_reply_content = ""
                description = ""
                tool = getattr(EDA_Tools, tool_name, None)
                if callable(tool):
                    num_columns = data.shape[1] - 1
                    col_exp_date = data.columns.tolist()
                    date_name = date_name_list[idx]
                    img_path, return_value = tool(data,taskid,idx,date_name)
                    await websocket.send_text("{}\n".format(description))
                    if return_value is not None:
                        results.append(return_value)
                    await websocket.send_text(f"\n## {tool_name} ")
                    if tool_name == "total_trend":
                        description = "总体趋势结果图："
                    if tool_name == "CBLOF":
                        description = "异常监测点图："
                    if tool_name == "fft_periodic":
                        description = "使用傅里叶变换分析周期性："
                    if len(img_path) != 0:
                        for img in img_path:
                            img_name = img.split("/")[-1]
                            oss_upload_by_path("science-images" ,"{}/{}/{}".format(user_name,taskid,img_name),img)
                            image_url=get_image_url("science-images","{}/{}/{}".format(user_name,taskid,img_name))
                            img_str = "\r\r![img]({})".format(image_url)
                            await websocket.send_text(img_str)
                            await websocket.send_text("\n")
                    
            
            if results is not None:
                df_desc = data.describe(include="all")
                prompt = self.PROMPT_TEMPLATE_STATISTIC.format(analysis=analysis, tool=tool_name, 
                        result=results, instruction=instruction)
                message = [self.llm._default_system_msg()]
                message.append(self.llm._user_msg(prompt))
                response = await llm.acompletion_text(message,temperature=0.7,timeout=3)
                collected_messages=[]
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
                    await websocket.send_text(chunk_message)
                collected_messages.append(chunk_message)
                return collected_messages

        return ""


class data_analysis_Input_Analyst(Role):
    name: str = "数据分析"
    profile: str = "数据分析、统计分析等, 擅长利用各种分析工具对数据进行科学分析。"

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([Input_Analysis])


async def start(
        idea: str = "",
        investment: float = 0,
        n_round: int = 1,
        add_human: bool = True,
):
    team = Team()
    team.hire(
        [
            data_analysis_Input_Analyst(),
        ]
    )

    team.run_project(idea)
    await team.run(n_round=n_round)


async def main():
    while True:
        userInput = input("\n\n老板，您好：")
        if userInput == "结束" or userInput == "exit":
            break
        else:
            await start(userInput)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
