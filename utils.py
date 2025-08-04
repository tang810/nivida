import requests
from dotenv import load_dotenv
import csv
import os
from pathlib import Path
import pandas as pd
import chardet
import json
from typing import Dict, Any, Tuple, Optional

# 加载 .env 文件
load_dotenv()
# 读取环境变量
server_base = os.getenv('server_base')


def get_file_preview(file_path: str, max_items: int = 5, max_str_length: int = 50) -> Dict[str, Any]:
    """
    获取文件预览信息，包括结构、字段名和部分数据
    
    Args:
        file_path: 文件路径
        max_items: 最多显示的数据项数量
        max_str_length: 字符串字段的最大显示长度
    
    Returns:
        包含文件预览信息的字典
    """
    path = Path(file_path)
    file_suffix = path.suffix
    preview = {
        "file_name": path.name,
        "file_type": file_suffix,
        "file_size": f"{os.path.getsize(file_path) / 1024:.1f} KB",
        "structure": {},
        "features": [],
        "sample_data": {},
        "error": None
    }
    
    try:
        if file_suffix == ".json":
            preview.update(get_json_preview(file_path, max_items, max_str_length))
        elif file_suffix == ".csv":
            preview.update(get_csv_preview(file_path, max_items, max_str_length))
        elif file_suffix == ".txt":
            preview.update(get_txt_preview(file_path, max_items, max_str_length))
        elif file_suffix in [".xlsx", ".xls"]:
            preview.update(get_excel_preview(file_path, max_items, max_str_length))
        else:
            preview["error"] = f"不支持的文件格式: {file_suffix}"
    except Exception as e:
        preview["error"] = f"文件预览失败: {str(e)}"
    
    return preview


def get_json_preview(file_path: str, max_items: int, max_str_length: int) -> Dict[str, Any]:
    """JSON文件预览"""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    
    preview = {}
    
    # 判断数据结构类型
    if isinstance(data, list):
        preview["structure"] = {
            "type": "数组",
            "length": len(data),
            "item_type": type(data[0]).__name__ if data else "空数组"
        }
        
        if data:
            # 获取特征信息
            if isinstance(data[0], dict):
                # 对象数组：收集所有可能的键
                all_keys = set()
                for item in data[:max_items]:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                preview["features"] = list(all_keys)
                
                # 样本数据
                preview["sample_data"] = []
                for i, item in enumerate(data[:max_items]):
                    if isinstance(item, dict):
                        sample_item = {}
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > max_str_length:
                                sample_item[key] = value[:max_str_length] + "..."
                            elif isinstance(value, list):
                                sample_item[key] = f"[列表，长度: {len(value)}]"
                            elif isinstance(value, dict):
                                sample_item[key] = f"{{对象，键数: {len(value)}}}"
                            else:
                                sample_item[key] = value
                        preview["sample_data"].append(sample_item)
            else:
                # 基础类型数组
                preview["features"] = ["value"]
                preview["sample_data"] = data[:max_items]
    
    elif isinstance(data, dict):
        preview["structure"] = {
            "type": "对象",
            "keys_count": len(data.keys()),
            "keys": list(data.keys())
        }
        preview["features"] = list(data.keys())
        
        # 样本数据
        sample_data = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_str_length:
                sample_data[key] = value[:max_str_length] + "..."
            elif isinstance(value, list):
                sample_data[key] = f"[列表，长度: {len(value)}，前几项: {value[:3]}]"
            elif isinstance(value, dict):
                sample_data[key] = f"{{对象，键: {list(value.keys())[:3]}}}"
            else:
                sample_data[key] = value
        preview["sample_data"] = sample_data
    
    else:
        preview["structure"] = {
            "type": type(data).__name__,
            "value": str(data)[:max_str_length]
        }
        preview["features"] = []
        preview["sample_data"] = str(data)[:max_str_length]
    
    return preview


def get_csv_preview(file_path: str, max_items: int, max_str_length: int) -> Dict[str, Any]:
    """CSV文件预览"""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    
    try:
        df = pd.read_csv(file_path, encoding=encoding, nrows=max_items)
        preview = {
            "structure": {
                "type": "表格数据",
                "rows": len(df),
                "columns": len(df.columns)
            },
            "features": df.columns.tolist(),
            "sample_data": []
        }
        
        # 获取样本数据
        for _, row in df.head(max_items).iterrows():
            sample_row = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    sample_row[col] = "NaN"
                elif isinstance(value, str) and len(value) > max_str_length:
                    sample_row[col] = value[:max_str_length] + "..."
                else:
                    sample_row[col] = value
            preview["sample_data"].append(sample_row)
        
        return preview
    except Exception as e:
        return {"error": f"CSV读取失败: {str(e)}"}


def get_txt_preview(file_path: str, max_items: int, max_str_length: int) -> Dict[str, Any]:
    """TXT文件预览"""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    
    try:
        # 先尝试作为分隔符文件读取
        with open(file_path, 'r', encoding=encoding) as file:
            first_lines = [file.readline().strip() for _ in range(min(10, max_items))]
        
        # 检测分隔符
        separators = [',', '\t', '|', ';', ' ']
        detected_sep = None
        max_splits = 0
        
        for sep in separators:
            splits = [len(line.split(sep)) for line in first_lines if line]
            if splits and len(set(splits)) == 1 and splits[0] > 1:
                if splits[0] > max_splits:
                    max_splits = splits[0]
                    detected_sep = sep
        
        if detected_sep:
            # 作为结构化数据处理
            df = pd.read_csv(file_path, encoding=encoding, sep=detected_sep, nrows=max_items)
            return get_csv_preview(file_path, max_items, max_str_length)
        else:
            # 作为纯文本处理
            with open(file_path, 'r', encoding=encoding) as file:
                lines = [file.readline().strip() for _ in range(max_items)]
            
            preview = {
                "structure": {
                    "type": "纯文本",
                    "lines": len([l for l in lines if l])
                },
                "features": ["text_content"],
                "sample_data": [line[:max_str_length] + ("..." if len(line) > max_str_length else "") 
                              for line in lines if line]
            }
            return preview
    except Exception as e:
        return {"error": f"TXT读取失败: {str(e)}"}


def get_excel_preview(file_path: str, max_items: int, max_str_length: int) -> Dict[str, Any]:
    """Excel文件预览"""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        preview = {
            "structure": {
                "type": "Excel工作簿",
                "sheets": sheet_names,
                "sheet_count": len(sheet_names)
            },
            "features": {},
            "sample_data": {}
        }
        
        # 预览每个工作表
        for sheet_name in sheet_names[:3]:  # 最多预览3个工作表
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_items)
                sheet_preview = {
                    "columns": df.columns.tolist(),
                    "rows": len(df),
                    "sample_data": []
                }
                
                # 获取样本数据
                for _, row in df.head(max_items).iterrows():
                    sample_row = {}
                    for col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            sample_row[col] = "NaN"
                        elif isinstance(value, str) and len(value) > max_str_length:
                            sample_row[col] = value[:max_str_length] + "..."
                        else:
                            sample_row[col] = value
                    sheet_preview["sample_data"].append(sample_row)
                
                preview["features"][sheet_name] = sheet_preview["columns"]
                preview["sample_data"][sheet_name] = sheet_preview["sample_data"]
            except Exception as e:
                preview["sample_data"][sheet_name] = f"工作表读取失败: {str(e)}"
        
        return preview
    except Exception as e:
        return {"error": f"Excel读取失败: {str(e)}"}


def format_preview_for_llm(preview: Dict[str, Any]) -> str:
    """
    将预览信息格式化为适合LLM阅读的文本
    """
    lines = []
    lines.append(f"=== 文件预览信息 ===")
    lines.append(f"文件名: {preview['file_name']}")
    lines.append(f"文件类型: {preview['file_type']}")
    lines.append(f"文件大小: {preview['file_size']}")
    
    if preview.get('error'):
        lines.append(f"错误信息: {preview['error']}")
        return "\n".join(lines)
    
    # 结构信息
    lines.append(f"\n--- 文件结构 ---")
    structure = preview.get('structure', {})
    for key, value in structure.items():
        lines.append(f"{key}: {value}")
    
    # 特征信息
    if preview.get('features'):
        lines.append(f"\n--- 数据特征/字段 ---")
        if isinstance(preview['features'], dict):
            for sheet, cols in preview['features'].items():
                lines.append(f"{sheet}: {cols}")
        else:
            lines.append(f"字段列表: {preview['features']}")
    
    # 样本数据
    if preview.get('sample_data'):
        lines.append(f"\n--- 样本数据 ---")
        sample_data = preview['sample_data']
        if isinstance(sample_data, dict) and any(isinstance(v, list) for v in sample_data.values()):
            # Excel格式或复杂结构
            for sheet, data in sample_data.items():
                if isinstance(data, list) and data:
                    lines.append(f"{sheet}工作表前几行:")
                    for i, row in enumerate(data[:3]):
                        lines.append(f"  行{i+1}: {row}")
        elif isinstance(sample_data, list):
            # 简单列表格式
            lines.append("前几行数据:")
            for i, row in enumerate(sample_data[:3]):
                lines.append(f"  行{i+1}: {row}")
        else:
            lines.append(f"样本: {sample_data}")
    
    return "\n".join(lines)


async def read_data_file(file_path: str, websocket=None):
    """
    读取数据文件，返回 (code, data, preview)
    code: 0成功, -1失败
    data: 成功时返回DataFrame，失败时返回错误信息
    preview: 文件预览信息，无论成功失败都返回
    """
    # 首先获取文件预览
    preview = get_file_preview(file_path)
    
    # 创建 Path 对象
    path = Path(file_path)
    # 提取文件扩展名
    file_suffix = path.suffix
    
    if file_suffix in [".json", ".txt", ".csv", ".xlsx", ".xls"]:
        try:
            if file_suffix == ".json":
                code, data = await read_json(file_path, websocket)
            elif file_suffix == ".csv":
                code, data = await read_csv(file_path, websocket)
            elif file_suffix == ".txt":
                code, data = await read_txt(file_path, websocket)
            elif file_suffix in [".xlsx", ".xls"]:
                code, data = await read_excel(file_path, websocket)
            
            # 无论成功失败，都返回预览信息
            return code, data, preview
            
        except Exception as e:
            return -1, f"文件处理异常: {str(e)}", preview
    else:
        return -1, "不允许的数据格式！", preview

async def check_and_add_time_column(df, websocket=None):
    """
    检查DataFrame是否有时间列，如果没有则添加索引列
    """
    # 检查是否存在时间字段
    time_columns = ["time", "date", "日期", "时间", "index"]
    has_time_field = any(col in df.columns for col in time_columns)
    
    if not has_time_field:
        message = "未检测到时间字段，正在添加索引列..."
        print(message)
        if websocket:
            await websocket.send_text(message + "\n")
        
        # 添加索引列
        df.insert(0, 'index', range(len(df)))
        print("已自动添加索引列作为时间维度")
    
    return df


async def read_json(file_path, websocket=None):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 读取JSON文件
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)

    # 检查JSON数据结构并进行适配
    if not isinstance(data, list):
        return -1, "JSON文件格式错误：数据应该是数组格式！"
    
    if len(data) == 0:
        return -1, "JSON文件为空！"

    # 检查数据结构类型
    first_item = data[0]
    
    # 情况1: 字符串数组或其他非字典类型，需要转换
    if not isinstance(first_item, dict):
        print("检测到非字典数组，正在转换为标准格式...")
        converted_data = []
        for idx, item in enumerate(data):
            # 创建字典格式，添加索引作为时间列
            if isinstance(item, str):
                converted_data.append({
                    "index": idx,  # 添加索引列作为时间维度
                    "value": item   # 原始数据作为值
                })
            elif isinstance(item, (int, float)):
                converted_data.append({
                    "index": idx,
                    "value": item
                })
            else:
                # 处理其他类型，转换为字符串
                converted_data.append({
                    "index": idx,
                    "value": str(item)
                })
        data = converted_data
        has_time_field = True  # 我们已经添加了index作为时间字段
        print("已自动添加索引列作为时间维度")
    
    # 情况2: 字典数组，检查是否有时间字段
    else:
        # 检查是否存在时间字段
        has_time_field = False
        for entry in data:
            if any(key in entry.keys() for key in ["time", "date", "日期", "时间"]):
                has_time_field = True
                break
        
        # 如果没有时间字段，添加索引列
        if not has_time_field:
            message = "未检测到时间字段，正在添加索引列..."
            print(message)
            # 通过websocket发送消息
            if websocket:
                await websocket.send_text(message + "\n")
                
            for idx, entry in enumerate(data):
                entry["index"] = idx
            has_time_field = True
            print("已自动添加索引列作为时间维度")

    # 处理数据并转换为所需的格式
    processed_data = []
    for entry in data:
        json_data_ = {}
        
        # 处理每个字段
        for key in entry.keys():
            # 处理时间字段
            if key in ["time", "date", "日期", "时间", "index"]:
                if isinstance(entry[key], str) and 'UTC=' in str(entry[key]):
                    time_str = str(entry[key]).replace('UTC=', '')
                    json_data_["date"] = time_str
                else:
                    json_data_["date"] = entry[key]
            else:
                # 处理非时间字段
                if isinstance(entry[key], list):
                    # 列表类型：展开为多列
                    for idx, data_ in enumerate(entry[key]):
                        json_data_["{}_{}".format(key, idx)] = data_
                elif isinstance(entry[key], (str, int, float)):
                    # 基础类型：直接使用
                    json_data_[key] = entry[key]
                else:
                    # 其他类型：转换为字符串
                    json_data_[key] = str(entry[key])
        
        processed_data.append(json_data_)

    # 将处理后的数据转换为DataFrame
    df = pd.DataFrame(processed_data)
    
    # 确保date列存在
    if "date" not in df.columns:
        return -1, "数据处理出错：无法创建时间列！"
    
    print(f"成功处理JSON文件，数据形状: {df.shape}")
    return 0, df


async def read_csv(file_path, websocket=None):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 使用检测到的编码格式读取CSV文件
    df = pd.read_csv(file_path, encoding=encoding)
    
    # 检查并添加时间列
    df = await check_and_add_time_column(df, websocket)
    
    return 0, df


async def read_txt(file_path, websocket=None):
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"检测到的编码格式: {encoding}")

    # 首先读取文件的前几行来检测分隔符
    with open(file_path, 'r', encoding=encoding) as file:
        sample_lines = [next(file) for _ in range(3)]  # 读取前3行用于检测
    
    # 常见的分隔符列表
    separators = [',', '\t', '|', ';', ' ']
    separator_counts = {}
    
    # 统计每种分隔符在样本行中的出现次数
    for sep in separators:
        count = sum(line.count(sep) for line in sample_lines)
        if count > 0:
            # 确保每行的分隔符数量一致
            splits = [len(line.split(sep)) for line in sample_lines]
            if len(set(splits)) == 1 and splits[0] > 1:  # 所有行的分隔数量相同且大于1
                separator_counts[sep] = count

    if not separator_counts:
        return -1, "无法检测到有效的分隔符"
    
    # 选择出现次数最多的分隔符
    detected_separator = max(separator_counts.items(), key=lambda x: x[1])[0]
    print(f"检测到的分隔符: {repr(detected_separator)}")

    try:
        # 使用检测到的分隔符读取文件
        df = pd.read_csv(file_path, 
                        encoding=encoding, 
                        sep=detected_separator,
                        skipinitialspace=True)  # 跳过分隔符后的空格
        
        # 检查并添加时间列
        df = await check_and_add_time_column(df, websocket)
        
        return 0, df
    except Exception as e:
        return -1, f"文件读取错误：{str(e)}"


async def read_excel(file_path, websocket=None):
    try:
        # 读取所有工作表
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) > 1:
            # 如果有多个工作表，读取第一个非空的工作表
            for sheet in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                if not df.empty:
                    print(f"使用工作表: {sheet}")
                    break
        else:
            # 只有一个工作表时直接读取
            df = pd.read_excel(file_path)
        
        # 处理合并单元格和空值
        df = df.fillna(method='ffill')
        
        # 检查并添加时间列
        df = await check_and_add_time_column(df, websocket)
        
        return 0, df
        
    except Exception as e:
        return -1, f"Excel文件读取错误：{str(e)}"


def upload_file(file_path, img_name, taskid, team_name):
    # FastAPI 服务器地址
    url = "{}/api/uploadBase".format(server_base)

    # 表单数据
    form_data = {
        "taskid": taskid,
        "team_name": team_name
    }

    # 读取文件内容
    with open(file_path, "rb") as file:
        # 发送 POST 请求
        files = [('files', (img_name, file))]
        print(files)
        response = requests.post(url, data=form_data, files=files)

    # 检查响应状态码
    if response.status_code == 200:
        print("文件上传成功:", response.json())
    else:
        print("文件上传失败:", response.text)