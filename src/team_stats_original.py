# team_stats.py
import io, os, warnings, json, time, re
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from alpha.logs import logger
from src.llm_utils import SeLLM, load_config
from src.storage_utils import StorageClient
from src.Tools.date_utils import check_date_column
from src.Tools.plot_utils import density_plot  # 若不需要密度图可移除
from src.Tools import EDA_Tools  # 用于 corr/fft 子模式
import chardet

MINIO_PUBLIC = os.getenv("MINIO_PUBLIC_ENDPOINT", "https://www.science42.tech")

# ---------- I/O 与工具通用函数 ----------
def _display_name(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("original_filename") or item.get("filename") or item.get("name") \
               or item.get("file_name") or os.path.basename(item.get("storage_key", "") or "") or "unknown"
    return os.path.basename(str(item))

def _looks_like_full_key(s: str) -> bool:
    return bool(re.match(r"^(upload|uploads|metadata|images|public|private)/", str(s).strip("/")))

def _extract_key(item: Any, user: str) -> str:
    if isinstance(item, dict):
        for k in ("storage_key","key","path"):
            v = item.get(k)
            if v: return str(v).lstrip("/")
        for k in ("original_filename","filename","name","file_name"):
            v = item.get(k)
            if v: return f"uploads/{user}/{v}"
        raise ValueError(f"Invalid file metadata: {item}")
    s = str(item).strip()
    return s.lstrip("/") if _looks_like_full_key(s) else f"uploads/{user}/{os.path.basename(s)}"

def _extract_bucket(item: Any, default="science-backend") -> str:
    return (item.get("bucket") if isinstance(item, dict) else None) or default

async def _read_df(file_bytes: bytes, filename: str) -> Tuple[int, Any, Dict[str, Any]]:
    try:
        encoding = (chardet.detect(file_bytes[:10000]) or {}).get("encoding") or "utf-8"
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".csv",".txt"):
            text = file_bytes.decode(encoding, errors="replace")
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                if df.shape[1] > 1:
                    return 0, df, {"size": len(file_bytes), "encoding": encoding, "shape": df.shape}
            except Exception:
                pass
            for sep in [",","\t",";","|"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if df.shape[1] > 1:
                        return 0, df, {"size": len(file_bytes), "encoding": encoding, "shape": df.shape}
                except Exception:
                    continue
        elif ext in (".xlsx",".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
            return 0, df, {"size": len(file_bytes), "encoding": "binary", "shape": df.shape}
        elif ext == ".json":
            text = file_bytes.decode(encoding, errors="replace")
            data = json.loads(text)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            return 0, df, {"size": len(file_bytes), "encoding": encoding, "shape": df.shape}
        return -1, "文件格式不支持或数据格式错误", {"size": len(file_bytes), "encoding": encoding}
    except Exception as e:
        return -1, f"文件读取失败: {e}", {"size": len(file_bytes), "error": str(e)}

async def _upload_png_bytes(img_bytes: bytes, user: str, taskid: str, name: str) -> str:
    storage = StorageClient()
    bucket_name = "science-images"
    key = f"{user}/{taskid}/{name}"
    await storage.aput_object(bucket_name, key, img_bytes, content_type="image/png")
    return f"{MINIO_PUBLIC.strip('/')}/{bucket_name}/{key}"

def _safe_str(obj: Any, max_chars=6000) -> str:
    s = str(obj)
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... (truncated)")

def _capture_df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO(); df.info(buf=buf); return buf.getvalue()

def _sanitize_llm_selection(sel: str) -> str:
    if not sel: return ""
    s = sel.strip()
    return re.sub(r'^[\[\(\s\'"]+|[\]\)\s\'"]+$', '', s)

def _auto_pick_date_col(df: pd.DataFrame) -> Optional[str]:
    # 简版兜底
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().sum() >= max(10, int(0.5*len(df))):
                return c
        except Exception:
            continue
    return None

# ---------- 主入口：统计分析 ----------
async def run_stats(websocket, user: str, taskid: str, file_metadata: list, instruction: str, submode: str = "stats"):
    await websocket.send_text(f"## 您输入的文件统计分析结果如下：\n\n")
    storage = StorageClient()

    # LLM
    cfg = load_config("config/config.yaml")
    llm = SeLLM(base_url=cfg["base_url_1"], api_key=cfg["api_key"])

    dfs: List[pd.DataFrame] = []
    infos: List[Dict[str, Any]] = []

    if not file_metadata:
        await websocket.send_text("未收到文件。\n"); return ""

    # 读取所有文件
    for item in file_metadata:
        bucket = _extract_bucket(item)
        key = _extract_key(item, user)
        try:
            content = await storage.aget_object(bucket, key)
            fname = _display_name(item)
            code, data_or_err, preview = await _read_df(content, fname)
            if code == -1:
                await websocket.send_text(f"- 文件 {fname} 读取失败：{data_or_err}\n")
                continue
            df: pd.DataFrame = data_or_err  # type: ignore
            dfs.append(df); infos.append({"filename": fname, "bucket": bucket, "key": key})
        except Exception as e:
            await websocket.send_text(f"- 读取 {bucket}/{key} 失败：{e}\n")

    if not dfs:
        await websocket.send_text("没有可用的数据集。\n"); return ""

    # 每个文件做统计概览；若 submode 为 corr/fft，则额外调用对应 EDA_Tools
    for i, df in enumerate(dfs):
        fname = infos[i]["filename"]
        await websocket.send_text(f"### 数据集：{fname}\n\n")
        await websocket.send_text(f"```\n{_safe_str(df.head())}\n```\n\n")

        info_text = _capture_df_info_text(df)
        prompt = f"""
你是统计分析专家。请基于下列数据概览给出简洁的中文表格化分析结论（不要输出代码）：

- 维度：{df.shape}
- 前五行：
{_safe_str(df.head())}
- 统计描述：
{_safe_str(df.describe(include='all'))}
- 结构信息：
{_safe_str(info_text)}

用户问题：{instruction}
只输出表格化要点（如变量类型、缺失、分布、可疑值等），简洁有层次。
""".strip()
        msgs = [llm._default_system_msg(), llm._user_msg(prompt)]
        stream = await llm.acompletion_text(msgs, temperature=0.4, timeout=30)
        async for c in stream:
            await websocket.send_text(c.choices[0].delta.content or "" if c.choices else "")
        await websocket.send_text("\n\n")

        # 可选：corr/fft 子模式
        try:
            nowstr = time.strftime("%Y%m%d%H%M%S")
            if submode == "corr":
                img_bufs, summary = EDA_Tools.acf_and_pacf(df, None)
                if img_bufs:
                    for k, buf in enumerate(img_bufs):
                        buf.seek(0)
                        url = await _upload_png_bytes(buf.getvalue(), user, taskid, f"acf_pacf_{os.path.splitext(fname)[0]}_{k}_{nowstr}.png")
                                                # 替换为 https 地址
                        if url.startswith(minio_addr):
                            url = url.replace(minio_addr, https_vip_addr, 1)
                        await websocket.send_text(f"![相关图#{k+1}]({url})\n\n")
                if summary:
                    await websocket.send_text(f"**相关性结论**：{summary}\n\n")

            if submode == "fft":
                img_bufs, summary = EDA_Tools.fft_periodic(df, None)
                if img_bufs:
                    for k, buf in enumerate(img_bufs):
                        buf.seek(0)
                        url = await _upload_png_bytes(buf.getvalue(), user, taskid, f"fft_{os.path.splitext(fname)[0]}_{k}_{nowstr}.png")
                                        # 替换为 https 地址
                        if url.startswith(minio_addr):
                            url = url.replace(minio_addr, https_vip_addr, 1)
                        await websocket.send_text(f"![频谱图#{k+1}]({url})\n\n")
                if summary:
                    await websocket.send_text(f"**频域结论**：{summary}\n\n")
        except Exception as e:
            await websocket.send_text(f"子模式({submode}) 生成图失败：{e}\n\n")

    return ""
