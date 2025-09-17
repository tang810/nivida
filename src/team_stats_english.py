# team_stats.py
import io, os, warnings, json, time, re
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from alpha.logs import logger
from src.llm_utils import SeLLM, load_config
from src.storage_utils import StorageClient
from src.Tools.date_utils import check_date_column
from src.Tools.plot_utils import density_plot  # Can be removed if density plots are not needed
from src.Tools import EDA_Tools  # Used for corr/fft submodes
import chardet

# ====== MinIO domain name (consistent with other modules) ======
MINIO_PUBLIC = os.getenv("MINIO_PUBLIC_ENDPOINT", "https://www.science42.tech").rstrip("/")
minio_addr   = os.getenv("MINIO_INTERNAL_ENDPOINT", "http://36.103.203.113:2300").rstrip("/")
https_vip_addr = MINIO_PUBLIC

# ---------- Common I/O and utility functions ----------
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
        return -1, "Unsupported file format or invalid data format", {"size": len(file_bytes), "encoding": encoding}
    except Exception as e:
        return -1, f"File reading failed: {e}", {"size": len(file_bytes), "error": str(e)}

async def _upload_png_bytes(img_bytes: bytes, user: str, taskid: str, name: str) -> str:
    storage = StorageClient()
    bucket_name = "science-images"
    key = f"{user}/{taskid}/{name}"
    await storage.aput_object(bucket_name, key, img_bytes, content_type="image/png")
    # Directly return the public https link
    return f"{MINIO_PUBLIC}/{bucket_name}/{key}"

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
    # Simple fallback
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().sum() >= max(10, int(0.5*len(df))):
                return c
        except Exception:
            continue
    return None

# ---------- Main entry: Statistical analysis ----------
async def run_stats(websocket, user: str, taskid: str, file_metadata: list, instruction: str, submode: str = "stats"):
    await websocket.send_text(f"## Statistical analysis results:\n\n")
    storage = StorageClient()

    # LLM (used for final summary; failure does not affect main process)
    llm = None
    try:
        cfg = load_config("config/config.yaml")
        llm = SeLLM(base_url=cfg["base_url_1"], api_key=cfg["api_key"])
    except Exception as e:
        await websocket.send_text(f"> LLM initialization failed (charts/tables will still be generated): {e}\n\n")

    dfs: List[pd.DataFrame] = []
    infos: List[Dict[str, Any]] = []

    if not file_metadata:
        await websocket.send_text("No files received.\n"); return ""

    # Read all files
    for item in file_metadata:
        bucket = _extract_bucket(item)
        key = _extract_key(item, user)
        try:
            content = await storage.aget_object(bucket, key)
            fname = _display_name(item)
            code, data_or_err, preview = await _read_df(content, fname)
            if code == -1:
                await websocket.send_text(f"- File {fname} failed to read: {data_or_err}\n")
                continue
            df: pd.DataFrame = data_or_err  # type: ignore
            dfs.append(df); infos.append({"filename": fname, "bucket": bucket, "key": key})
        except Exception as e:
            await websocket.send_text(f"- Failed to read {bucket}/{key}: {e}\n")

    if not dfs:
        await websocket.send_text("No available datasets.\n"); return ""

    # Statistical overview for each file; if submode is corr/fft, call corresponding EDA_Tools
    for i, df in enumerate(dfs):
        fname = infos[i]["filename"]
        await websocket.send_text(f"### Dataset: {fname}\n\n")
        await websocket.send_text(f"```\n{_safe_str(df.head())}\n```\n\n")

        # ====== Basic statistical tabular interpretation (generated by LLM) ======
        info_text = _capture_df_info_text(df)
        prompt = f"""
You are a statistical analysis expert. Based on the following dataset overview, provide a concise tabular analysis in English (do not output code):

- Shape: {df.shape}
- First five rows:
{_safe_str(df.head())}
- Statistical description:
{_safe_str(df.describe(include='all'))}
- Structural info:
{_safe_str(info_text)}

User question: {instruction}
Only output tabular key points (such as variable types, missing values, distributions, suspicious values, etc.), concise and well-structured.
""".strip()
        if llm:
            msgs = [llm._default_system_msg(), llm._user_msg(prompt)]
            stream = await llm.acompletion_text(msgs, temperature=0.4, timeout=30)
            async for c in stream:
                await websocket.send_text(c.choices[0].delta.content or "" if c.choices else "")
            await websocket.send_text("\n\n")
        else:
            await websocket.send_text("> LLM unavailable, skipping tabular interpretation.\n\n")

        # ====== Optional: corr/fft submode ======
        try:
            nowstr = time.strftime("%Y%m%d%H%M%S")
            if submode == "corr":
                img_bufs, summary = EDA_Tools.acf_and_pacf(df, None)
                if img_bufs:
                    for k, buf in enumerate(img_bufs):
                        buf.seek(0)
                        url = await _upload_png_bytes(buf.getvalue(), user, taskid, f"acf_pacf_{os.path.splitext(fname)[0]}_{k}_{nowstr}.png")
                        if url.startswith(minio_addr):
                            url = url.replace(minio_addr, https_vip_addr, 1)
                        await websocket.send_text(f"![Correlation plot #{k+1}]({url})\n\n")
                if summary:
                    await websocket.send_text(f"**Correlation conclusion**: {summary}\n\n")

            if submode == "fft":
                img_bufs, summary = EDA_Tools.fft_periodic(df, None)
                if img_bufs:
                    for k, buf in enumerate(img_bufs):
                        buf.seek(0)
                        url = await _upload_png_bytes(buf.getvalue(), user, taskid, f"fft_{os.path.splitext(fname)[0]}_{k}_{nowstr}.png")
                        if url.startswith(minio_addr):
                            url = url.replace(minio_addr, https_vip_addr, 1)
                        await websocket.send_text(f"![Spectrum plot #{k+1}]({url})\n\n")
                if summary:
                    await websocket.send_text(f"**Frequency domain conclusion**: {summary}\n\n")
        except Exception as e:
            await websocket.send_text(f"Submode({submode}) failed to generate plots: {e}\n\n")

        # ====== Final step: LLM overall summary (concise, no repetition of tables) ======
        if llm:
            try:
                # Prepare a brief "key points package" for summary
                na_rates = (df.isna().mean()*100.0).sort_values(ascending=False)
                top_missing = na_rates.head(5).round(2).to_dict()
                col_types = df.dtypes.astype(str).to_dict()
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                cat_cols     = df.select_dtypes(exclude=["number"]).columns.tolist()

                summary_payload = {
                    "shape": list(df.shape),
                    "numeric_cols_count": len(numeric_cols),
                    "categorical_cols_count": len(cat_cols),
                    "top_missing(%)": top_missing
                }

                sys_prompt = (
                    "You are a senior data analyst. Based on the given statistics, output a concise summary in English within **500 words**; "
                    "Do not use emojis; do not simply repeat previous tables or literal fields; "
                    "Structure the output as 'Overall overview / Data quality & risks / Suggestions', short and powerful."
                )
                user_prompt = f"""User question: {instruction}
File: {fname}
Key points: {json.dumps(summary_payload, ensure_ascii=False)}
(If there are 'Correlation/Frequency domain conclusions', please integrate them into the final suggestions as well.)"""

                msgs = [
                    llm._default_system_msg(),
                    llm._user_msg(sys_prompt + "\n\n" + user_prompt),
                ]

                await websocket.send_text("### Statistical interpretation:\n\n")
                char_limit = 1000  # keep same as trend
                produced = ""
                stream = await llm.acompletion_text(msgs, temperature=0.6, timeout=60)
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else ""
                    if not delta:
                        continue
                    remain = char_limit - len(produced)
                    if remain <= 0:
                        break
                    out = delta[:remain]
                    produced += out
                    await websocket.send_text(out)
                await websocket.send_text("\n\n")
            except Exception as e:
                await websocket.send_text(f"> Failed to generate text summary: {e}\n\n")

    return ""
