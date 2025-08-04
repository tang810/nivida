import re
import pandas as pd

def check_date_column(date_name: str, df: pd.DataFrame):
    """检查和处理数据框中的日期列
    
    Args:
        date_name (str): 用户指定的日期列名或索引,如果为空字符串则表示未指定
        df (pd.DataFrame): 待处理的数据框
    
    Returns:
        tuple: (是否成功, 日期列名, 错误信息) (success, col_name, error_msg)
    """
    def try_convert_datetime(col_name):
        try:
            # 检查是否为数值类型
            if pd.api.types.is_numeric_dtype(df[col_name]):
                # 如果是整数类型，将其视为有效的时间序列索引
                if pd.api.types.is_integer_dtype(df[col_name]):
                    return True, ""
                return False, f"列 '{col_name}' 包含浮点数而不是日期格式或整数索引，请指定正确的列。"
            
            # 尝试转换为日期格式
            converted_series = pd.to_datetime(df[col_name])
            
            # 更新DataFrame中的列
            df[col_name] = converted_series
            return True, ""
            
        except Exception as e:
            return False, f"列 '{col_name}' 无法转换为日期格式，请检查数据格式后重新上传。错误信息: {str(e)}。"
    
    # 如果用户指定了日期列
    if date_name:
        # 检查列名是否存在
        if date_name not in df.columns:
            # 尝试将输入解释为列索引
            try:
                col_idx = int(date_name) - 1  # 用户输入的列号从1开始
                if 0 <= col_idx < len(df.columns):
                    date_name = df.columns[col_idx]
                else:
                    return False, "", f"指定的列索引 {date_name} 超出范围。"
            except ValueError:
                return False, "", f"找不到名为 '{date_name}' 的列。"
        
        # 尝试转换为datetime或验证整数索引
        success, error_msg = try_convert_datetime(date_name)
        if success:
            return True, date_name, ""
        return False, "", error_msg
    
    # 如果用户未指定日期列,检查第一列
    # first_col = df.columns[0]
    # success, error_msg = try_convert_datetime(first_col)
    # if success:
    #     return True, first_col, ""
    
    # 第一列不是日期列且用户未指定日期列
    return False, "", "数据中无日期列或索引列，且用户未指定。"
