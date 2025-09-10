# team_anomaly.py
import os, io, json, time, re
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple
from alpha.logs import logger
from src.storage_utils import StorageClient
from src.Tools import EDA_Tools
import chardet

MINIO_PUBLIC = os.getenv("MINIO_PUBLIC_ENDPOINT", "https://www.science42.tech")

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

async def _read_df(file_bytes: bytes, filename: str):
    import pandas as pd, io, chardet, json, os
    try:
        encoding = (chardet.detect(file_bytes[:10000]) or {}).get("encoding") or "utf-8"
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".csv",".txt"):
            text = file_bytes.decode(encoding, errors="replace")
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                if df.shape[1] > 1: return 0, df
            except Exception: pass
            for sep in [",","\t",";","|"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if df.shape[1] > 1: return 0, df
                except Exception: continue
        elif ext in (".xlsx",".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes)); return 0, df
        elif ext == ".json":
            text = file_bytes.decode(encoding, errors="replace")
            data = json.loads(text)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            return 0, df
        return -1, "unsupported"
    except Exception as e:
        return -1, str(e)

async def _upload_png_bytes(img_bytes: bytes, user: str, taskid: str, name: str) -> str:
    storage = StorageClient()
    bucket_name = "science-images"
    key = f"{user}/{taskid}/{name}"
    await storage.aput_object(bucket_name, key, img_bytes, content_type="image/png")
    return f"{MINIO_PUBLIC.strip('/')}/{bucket_name}/{key}"

async def run_anomaly(websocket, user: str, taskid: str, file_metadata: list, instruction: str):
    await websocket.send_text("## 异常检测（CBLOF）\n\n")
    storage = StorageClient()
    if not file_metadata:
        await websocket.send_text("未收到文件。\n"); return ""

    for item in file_metadata:
        bucket = _extract_bucket(item)
        key = _extract_key(item, user)
        try:
            content = await storage.aget_object(bucket, key)
            fname = _display_name(item)
            code, df_or_err = await _read_df(content, fname)
            if code == -1:
                await websocket.send_text(f"- {fname} 读取失败：{df_or_err}\n"); continue
            df: pd.DataFrame = df_or_err  # type: ignore

            img_bufs, summary = EDA_Tools.CBLOF(df, None)  # CBLOF 通常不强制时间列
            nowstr = time.strftime("%Y%m%d%H%M%S")
            uploaded = False
            if img_bufs:
                for k, buf in enumerate(img_bufs):
                    buf.seek(0)
                    url = await _upload_png_bytes(buf.getvalue(), user, taskid, f"cblof_{os.path.splitext(fname)[0]}_{k}_{nowstr}.png")
                    await websocket.send_text(f"![异常图#{k+1}]({url})\n\n")
                    uploaded = True
            if not uploaded:
                await websocket.send_text(f"- {fname} 未返回异常图。\n")
            if summary:
                await websocket.send_text(f"**异常结论**：{summary}\n\n")

        except Exception as e:
            await websocket.send_text(f"- 处理 {bucket}/{key} 失败：{e}\n")

    return ""
