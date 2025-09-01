# === team_config.py (MinIO + BytesIO，简化日志版本) ===
import os
import re
import io
import sys
import json
import time
import chardet
from typing import ClassVar, Optional, Any, Dict, List, Tuple, Union, Pattern
import re

import pandas as pd

from alpha.team import Team
from alpha.roles import Role
from alpha.logs import logger
from alpha.actions import Action, UserRequirement

from dotenv import load_dotenv

# 仅保留原依赖（如需）
from langchain_community.vectorstores import Chroma  # noqa: F401
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings  # noqa: F401

# 你的 LLM 与工具（假设 EDA_Tools 已按 BytesIO 版本改造）
from src.llm_utils import SeLLM, load_config
from src.Tools import EDA_Tools
from src.Tools.date_utils import check_date_column
from src.Tools.plot_utils import density_plot  # 其余图按需引入（均应支持 BytesIO）

# MinIO 客户端（boto3 封装）
from src.storage_utils import StorageClient

# 若有你自己的格式工具
from utils import format_preview_for_llm

load_dotenv()

# 简化日志配置 - 只记录ERROR级别
handler = {"sink": sys.stdout, "level": "ERROR"}
logger.configure(handlers=[handler])

init_file_metadata: Optional[List[Any]] = None
df_data_list: Optional[List[pd.DataFrame]] = None


# ---------------------------- 小工具 ----------------------------

def _capture_df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def _safe_str(obj: Any, max_chars: int = 6000) -> str:
    s = str(obj)
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... (truncated)")


def _looks_like_full_key(s: str) -> bool:
    return bool(re.match(r"^(uploads|metadata|images|public|private)/", s.strip("/")))


async def _read_dataframe_from_memory(file_content: bytes, filename: str) -> Tuple[int, Union[pd.DataFrame, str], Dict[str, Any]]:
    """
    返回:
      code = 0: 成功, data 是 DataFrame
      code = -1: 失败, data 是错误信息字符串
      preview: {size, encoding, shape? 或 error?}
    """
    try:
        encoding = (chardet.detect(file_content[:10000]) or {}).get("encoding") or "utf-8"
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".csv", ".txt"):
            text = file_content.decode(encoding, errors="replace")
            # 先尝试自动推断
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                if df.shape[1] > 1:
                    return 0, df, {"size": len(file_content), "encoding": encoding, "shape": df.shape}
            except Exception:
                pass
            # 常见分隔符回退
            for sep in [",", "\t", ";", "|"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if df.shape[1] > 1:
                        return 0, df, {"size": len(file_content), "encoding": encoding, "shape": df.shape}
                except Exception:
                    continue

        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_content))
            return 0, df, {"size": len(file_content), "encoding": "binary", "shape": df.shape}

        elif ext == ".json":
            text = file_content.decode(encoding, errors="replace")
            data = json.loads(text)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            return 0, df, {"size": len(file_content), "encoding": encoding, "shape": df.shape}

        return -1, "文件格式不支持或数据格式错误", {"size": len(file_content), "encoding": encoding}
    except Exception as e:
        return -1, f"文件读取失败: {e}", {"size": len(file_content), "error": str(e)}


async def _analyze_failed_file_with_llm(websocket, llm: SeLLM, file_preview: Dict[str, Any], error_msg: str, instruction: str):
    prompt = f"""
你是数据分析专家。用户上传了一个文件，但文件读取失败了。
不过我获取了一些文件的基本信息，请你基于这些信息进行分析和建议。

用户的问题是：{instruction}

文件错误信息：{error_msg}

文件基本信息：
{format_preview_for_llm(file_preview)}

请你：
1. 分析文件内容和结构
2. 解释为什么文件读取失败
3. 提供具体的修复建议
4. 如果可能，推测用户想要进行什么样的分析

请给出详细且有用的分析和建议：
""".strip()

    msgs = [llm._default_system_msg(), llm._user_msg(prompt)]
    stream = await llm.acompletion_text(msgs, temperature=0.7, timeout=3)

    await websocket.send_text("## 文件分析和建议\n\n")
    async for chunk in stream:
        chunk_msg = chunk.choices[0].delta.content or "" if chunk.choices else ""
        await websocket.send_text(chunk_msg)
    await websocket.send_text("\n\n---\n\n")


# ---------------------------- 主 Action ----------------------------

class Input_Analysis(Action):
    name: str = "基本信息解读"
    desc: str = "数据文件基本信息解读"

    # 新增：域名配置需要标注为 ClassVar
    minio_internal: ClassVar[str] = os.getenv("MINIO_INTERNAL_ENDPOINT", "http://36.103.203.113:2300")
    minio_public: ClassVar[str] = os.getenv("MINIO_PUBLIC_ENDPOINT", "https://www.science42.tech")

    # 原来这里是：_illegal_key_pattern = re.compile(r"...")
    # 改为 ClassVar + Pattern 注解，避免被当作模型字段
    _illegal_key_pattern: ClassVar[Pattern[str]] = re.compile(r"[{}'\"\\^`<>?]")

    # 如果你有这个变量，已是 ClassVar（保留即可）
    data: ClassVar[str] = None

    # 这三个模板原来没有类型注解；加上 ClassVar[str]
    PROMPT_TEMPLATE: ClassVar[str] = """
你是统计分析专家。你需要根据输入数据的 python 统计量对输入数据进行初步描述、解释以及分析：

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

请给出你的详细分析, 结果仅以表格形式输出：
""".strip()

    PROMPT_TEMPLATE_EXTRACT_DATE: ClassVar[str] = """
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

请注意，无需其他描述性文字，直接给出列名或者索引或者空结果，加上中括号[]，即返回值是[列名]、[索引]、[]这三者之一。
如果用户指定第几列是日期列，你需要直接返回该列的位置，而不是列名，即[列索引]。
如果日期、时间等列与年份、月份等同时存在，优先选择名为日期、时间的列作为时间戳。
""".strip()

    PROMPT_TEMPLATE_STATISTIC: ClassVar[str] = """
你是数据科学家，擅长统计分析，你需要按照以下步骤一步步分析，逻辑清晰，不要输出任何代码，请使用中文分析：
1、结合数据分析结果{analysis}，对结果{result}进行专业的综合性分析与解读

请给出你的回答：
""".strip()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage: Optional[StorageClient] = None

    # ---- MinIO ----
    def _ensure_storage(self):
        if self.storage is None:
            self.storage = StorageClient()

    @staticmethod
    def _display_name(item: Union[str, Dict[str, Any]]) -> str:
        if isinstance(item, dict):
            return (
                item.get("original_filename")
                or item.get("filename")
                or item.get("name")
                or item.get("file_name")
                or os.path.basename(item.get("storage_key", "") or "")
                or "unknown"
            )
        return os.path.basename(str(item))

    def _extract_key(self, item: Union[str, Dict[str, Any]], user: str) -> str:
        if isinstance(item, dict):
            for k in ("storage_key", "key", "path"):
                v = item.get(k)
                if v:
                    return str(v).lstrip("/")
            for k in ("original_filename", "filename", "name", "file_name"):
                v = item.get(k)
                if v:
                    return f"upload/{user}/{v}"
            raise ValueError(f"Invalid file metadata: {item}")
        s = str(item).strip()
        return s.lstrip("/") if _looks_like_full_key(s) else f"upload/{user}/{os.path.basename(s)}"

    def _extract_bucket(self, item: Union[str, Dict[str, Any]], default: str = "science-backend") -> str:
        return (item.get("bucket") if isinstance(item, dict) else None) or default

    async def _get_pres_url(self, bucket: str, key: str) -> str:
        """
        生成预签名URL并替换为HTTPS域名
        """
        self._ensure_storage()
        try:
            # 生成预签名URL
            url = self.storage.generate_presigned_url(bucket, key, expires_in=7 * 24 * 3600)
            
            # 替换域名
            if url.startswith(self.minio_internal):
                url = url.replace(self.minio_internal, self.minio_public, 1)
            
            return url
        except Exception as e:
            logger.error(f"Failed to generate/convert URL for {bucket}/{key}: {e}")
            raise

    async def _upload_png_bytes(self, img_bytes: bytes, user: str, taskid: str, name: str) -> str:
        """
        上传PNG图片字节到MinIO并返回【永久的公开访问URL】
        如果上传或URL生成失败，直接抛出异常
        """
        self._ensure_storage()
        bucket_name = "science-images" # 定义存储桶名称
        key = f"{user}/{taskid}/{name}"
        
        try:
            # 1. 上传图片到MinIO
            await self.storage.aput_object(bucket_name, key, img_bytes, content_type="image/png")
            
            # 2. 【核心修改】直接拼接永久公开URL，不再调用 _get_pres_url
            #    URL格式: <公网访问域名>/<存储桶名>/<对象路径>
            #    self.minio_public 来自您的类变量定义，例如 "https://www.science42.tech"
            url = f"{self.minio_public.strip('/')}/{bucket_name}/{key}"
            
            return url
            
        except Exception as e:
            # 日志信息可以更具体一些
            logger.error(f"Failed to upload or generate public URL for {key}: {e}")
            raise

    # ---- 主流程 ----
    async def run(self, instruction: str, *args):
        websocket = args[0]
        user, taskid, file_metadata = args[1], args[2], args[3]

        # LLM
        cfg = load_config("config/config.yaml")
        llm = SeLLM(base_url=cfg["base_url_1"], api_key=cfg["api_key"])

        global df_data_list, init_file_metadata, date_name_list
        df_data_list, date_name_list = [], []
        init_file_metadata = file_metadata

        # 逐个对象：MinIO -> bytes -> DataFrame
        loaded_infos: List[Dict[str, Any]] = []
        for idx, item in enumerate(file_metadata, 1):
            try:
                bucket = self._extract_bucket(item)
                key = self._extract_key(item, user)
                if self._illegal_key_pattern.search(key):
                    raise ValueError(f"Illegal chars in key: {key}")

                self._ensure_storage()
                content = await self.storage.aget_object(bucket, key)
                fname = self._display_name(item)

                code, data_or_err, preview = await _read_dataframe_from_memory(content, fname)
                if code == -1:
                    await websocket.send_text(f"文件 {fname} 读取失败，让我分析一下...\n\n")
                    await websocket.send_text(f"```\n{format_preview_for_llm(preview)}\n```\n\n")
                    await _analyze_failed_file_with_llm(websocket, llm, preview, str(data_or_err), instruction)
                    continue

                df: pd.DataFrame = data_or_err  # type: ignore
                df_data_list.append(df)
                loaded_infos.append({"filename": fname, "bucket": bucket, "key": key})
            except Exception as e:
                logger.error(f"Failed to load file {self._display_name(item)}: {e}")

        if not df_data_list:
            await websocket.send_text("没有找到可读取的数据文件。")
            return ""

        # 提取日期列
        for df in df_data_list:
            head, shape, desc = df.head(), df.shape, df.describe(include="all")
            info_text = _capture_df_info_text(df)
            prompt = self.PROMPT_TEMPLATE_EXTRACT_DATE.format(
                instruction=instruction,
                head=_safe_str(head),
                shape=str(shape),
                desc=_safe_str(desc),
                info=_safe_str(info_text),
            )
            msgs = [llm._default_system_msg(), llm._user_msg(prompt)]
            stream = await llm.acompletion_text(msgs, temperature=0.7, timeout=3)

            chunks: List[str] = []
            async for c in stream:
                chunks.append(c.choices[0].delta.content or "" if c.choices else "")
            raw = "".join(chunks).strip()
            sel = raw[1:-1].strip() if raw.startswith("[") and raw.endswith("]") else raw

            ok, col_name, err = check_date_column(sel, df)
            date_name_list.append(col_name if ok else None)
            if not ok and err:
                await websocket.send_text(err)

        # 分析 + 绘图（内存）
        for i, df in enumerate(df_data_list):
            fname = loaded_infos[i]["filename"] if i < len(loaded_infos) else f"dataset_{i}"
            await websocket.send_text(f"## 分析 {fname} 数据集\n\n")
            await websocket.send_text(f"```\n{_safe_str(df.head())}\n```\n\n")

            nowstr = time.strftime("%Y%m%d%H%M%S", time.localtime())

            # 基础分析 -> LLM
            info_text = _capture_df_info_text(df)
            prompt = self.PROMPT_TEMPLATE.format(
                instruction=instruction,
                head=_safe_str(df.head()),
                shape=str(df.shape),
                desc=_safe_str(df.describe(include="all")),
                info=_safe_str(info_text),
            )
            msgs = [llm._default_system_msg(), llm._user_msg(prompt)]
            stream = await llm.acompletion_text(msgs, temperature=0.7, timeout=3)

            collected: List[str] = []
            async for c in stream:
                msg = c.choices[0].delta.content or "" if c.choices else ""
                await websocket.send_text(msg)
                collected.append(msg)
            await websocket.send_text("\n\n")
            analysis_text = "".join(collected)

            # 密度图（BytesIO -> 上传 -> URL）
            await websocket.send_text(f"### {fname} 密度分布图\n\n")
            try:
                buf = io.BytesIO()
                density_plot(df, buf)  # 要求 plot_utils 支持 BytesIO
                buf.seek(0)
                url = await self._upload_png_bytes(buf.getvalue(), user, taskid, f"density_{os.path.splitext(fname)[0]}_{nowstr}.png")
                await websocket.send_text(f"![密度图]({url})\n\n")
                buf.close()
            except Exception as e:
                await websocket.send_text(f"密度图生成失败: {e}\n\n")

            # EDA 工具（统一返回 (List[BytesIO], summary)）
            tools = ["acf_and_pacf", "total_trend", "fft_periodic", "CBLOF"]  # 确保这个变量被正确定义
            summaries: List[Any] = []
            date_col = date_name_list[i]

            for t in tools:
                fn = getattr(EDA_Tools, t, None)
                if not callable(fn):
                    await websocket.send_text(f"### {t} 工具不可用\n\n")
                    continue
                    
                await websocket.send_text(f"### {t} 分析结果\n\n")
                try:
                    # 修复: 调用正确的函数，不是总调用total_trend
                    img_bufs, summary = fn(df, date_col)
                    summaries.append(summary)
                    
                    # 处理图片上传
                    if img_bufs:
                        for idx, buf in enumerate(img_bufs):
                            buf.seek(0)
                            # 修复: 定义filename变量
                            img_filename = f"{t}_{os.path.splitext(fname)[0]}_{idx}_{nowstr}.png"
                            try:
                                url = await self._upload_png_bytes(buf.getvalue(), user, taskid, img_filename)
                                await websocket.send_text(f"![{t}图]({url})\n\n")
                            except Exception as e:
                                await websocket.send_text(f"{t}图片上传失败: {e}\n\n")
                            finally:
                                buf.close()
                    
                    # 输出分析结果文本
                    if summary:
                        await websocket.send_text(f"**分析结果**: {summary}\n\n")
                        
                except Exception as e:
                    await websocket.send_text(f"{t} 分析失败: {e}\n\n")
                    logger.error(f"EDA tool {t} error: {e}")

            # 综合结论
            if summaries:
                # 过滤掉空的summary
                valid_summaries = [s for s in summaries if s and s.strip()]
                if valid_summaries:
                    prompt2 = self.PROMPT_TEMPLATE_STATISTIC.format(
                        analysis=_safe_str(analysis_text, 4000),
                        result=_safe_str(valid_summaries, 4000),
                    )
                    msgs = [llm._default_system_msg(), llm._user_msg(prompt2)]
                    stream = await llm.acompletion_text(msgs, temperature=0.7, timeout=3)
                    await websocket.send_text("### 综合分析结论\n\n")
                    async for c in stream:
                        await websocket.send_text(c.choices[0].delta.content or "" if c.choices else "")
                    await websocket.send_text("\n\n")


# ---------------------------- 角色/入口 ----------------------------

class data_analysis_Input_Analyst(Role):
    name: str = "数据分析"
    profile: str = "数据分析、统计分析等, 擅长利用各种分析工具对数据进行科学分析。"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([Input_Analysis])


async def start(idea: str = "", investment: float = 0, n_round: int = 1, add_human: bool = True):
    team = Team()
    team.hire([data_analysis_Input_Analyst()])
    team.run_project(idea)
    await team.run(n_round=n_round)


async def main():
    while True:
        userInput = input("\n\n老板，您好：")
        if userInput in ("结束", "exit"):
            break
        else:
            await start(userInput)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())