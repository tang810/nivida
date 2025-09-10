# src/team_router.py
import os
import re
import json
import time
import types
import importlib
import traceback
from typing import Any, Dict, List, Optional, Tuple

from alpha.roles import Role
from alpha.actions import Action, UserRequirement

# ---------------------- Debug Switch ----------------------
DEBUG = os.getenv("ROUTER_DEBUG", "1") != "0"


# ---------------------- Utils ----------------------
async def _ws_log(websocket, msg: str):
    """统一 websocket + stdout 日志"""
    try:
        if DEBUG:
            print(msg, flush=True)
        await websocket.send_text(msg + ("\n\n" if not msg.endswith("\n") else ""))
    except Exception:
        try:
            if DEBUG:
                print("[router-log-failed]", msg, flush=True)
        except Exception:
            pass


def _mask(s: Optional[str]) -> str:
    if not s:
        return "<unset>"
    if len(s) <= 4:
        return "***"
    return "***" + s[-4:]


def _summarize_files(file_metadata: Any) -> str:
    if not isinstance(file_metadata, (list, tuple)) or not file_metadata:
        return "files: 0"
    lines = [f"files: {len(file_metadata)}"]
    show_keys = ("bucket", "storage_key", "key", "path", "original_filename", "filename", "name", "file_name")
    for i, it in enumerate(file_metadata[:3], 1):
        if isinstance(it, dict):
            brief = {k: it.get(k) for k in show_keys if it.get(k) is not None}
            for k, v in list(brief.items()):
                if isinstance(v, str) and len(v) > 80:
                    brief[k] = "..." + v[-77:]
            lines.append(f"  [{i}] {json.dumps(brief, ensure_ascii=False)}")
        else:
            s = str(it)
            if len(s) > 80:
                s = "..." + s[-77:]
            lines.append(f"  [{i}] {s}")
    if len(file_metadata) > 3:
        lines.append(f"  ... (+{len(file_metadata) - 3} more)")
    return "\n".join(lines)


def _lazy_import(module_name: str, func_name: str) -> Tuple[Any, types.ModuleType]:
    """
    总是从 src.* 导入；并对 team_anomaly / team_anamoly 双拼写兜底。
    返回 (fn, module)
    """
    tried: List[str] = []

    def _try(mn: str):
        tried.append(mn)
        mod = importlib.import_module(mn)
        return getattr(mod, func_name), mod

    base = module_name if module_name.startswith("src.") else f"src.{module_name}"
    cand_modules = [base]
    if base.endswith(".team_anomaly"):
        cand_modules.append(base.replace("team_anomaly", "team_anamoly"))
    if base.endswith(".team_anamoly"):
        cand_modules.append(base.replace("team_anamoly", "team_anomaly"))

    last_err = None
    for mn in cand_modules:
        try:
            fn, mod = _try(mn)
            return fn, mod
        except Exception as e:
            last_err = e
    raise ImportError(f"Cannot import {module_name}.{func_name}; tried={tried}; last_err={last_err}")


def _extract_text_role(obj: Any) -> Tuple[str, Optional[str]]:
    if hasattr(obj, "content"):
        try:
            text = getattr(obj, "content")
            role = getattr(obj, "role", None)
            if isinstance(text, str):
                return text, (role if isinstance(role, str) else None)
        except Exception:
            pass

    if isinstance(obj, dict):
        for k in ("text", "content", "value", "message"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v, (obj.get("role") if isinstance(obj.get("role"), str) else None)

    if isinstance(obj, str):
        return obj, None

    return "", None


def _norm_text(instruction: Any, debug_log: List[str]) -> str:
    """
    只取“当前这一条”的最后一条用户消息做匹配；
    main.py 已改为只传 [{"text": idea, "role": "user"}]，这里仍做健壮处理。
    """
    candidates: List[Tuple[str, Optional[str]]] = []

    if isinstance(instruction, (list, tuple)):
        for x in reversed(instruction):
            txt, role = _extract_text_role(x)
            if txt and txt.strip():
                candidates.append((txt, role))
    else:
        txt, role = _extract_text_role(instruction)
        if txt and txt.strip():
            candidates.append((txt, role))

    picked = ""
    for txt, role in candidates:
        if role and str(role).lower() in {"user", "human"}:
            picked = txt
            break
    if not picked and candidates:
        picked = candidates[0][0]

    debug_log.append(f"[router] 候选消息条数: {len(candidates)}")
    if candidates:
        preview = (picked[:200] + "..." if len(picked) > 200 else picked)
        debug_log.append(f"[router] 选用文本预览: {preview!r}")

    # 去噪归一化
    s = picked
    m = re.search(r"===\s*当前用户问题.*?===.*?(?:用户|user)\s*[:：]\s*(.+)$", s, flags=re.I | re.S)
    if m:
        s = m.group(1)

    s = re.sub(r"===\s*历史聊天记录.*?(?====|$)", " ", s, flags=re.I | re.S)
    s = re.sub(r"`{3,}.*?`{3,}", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
    s = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"^\s*(human|assistant|system)\s*:\s*", "", s, flags=re.I)

    s = s.replace("\u3000", " ")
    for ch in "，。；：、！·（）【】《》“”":
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _compile_kw(k: str) -> re.Pattern:
    if re.search(r"[a-z]", k, re.I):
        return re.compile(rf"(?<![a-z]){re.escape(k)}(?![a-z])", re.I)
    return re.compile(re.escape(k))


def _hit_any(text: str, kws: List[str]) -> Optional[str]:
    for k in kws:
        if _compile_kw(k).search(text):
            return k
    return None


def _has_files(file_metadata: Any) -> bool:
    return isinstance(file_metadata, (list, tuple)) and len(file_metadata) > 0


# ---------------------- Action ----------------------
class RouteAction(Action):
    name: str = "路由分析"
    desc: str = "根据提示词选择对应分析模块；未命中则走通用LLM对话"

    async def run(self, instruction: Any, *args):
        websocket, user_name, taskid, file_metadata = args[0], args[1], args[2], args[3]

        # 强标记：一进入就输出，便于确认确实走进路由器
        header = (
            "ROUTER_OK\n"
            f"> router build=v6 | file={__file__}\n"
            f"> pid={os.getpid()} | cwd={os.getcwd()} | python={os.sys.executable}\n"
        )
        await _ws_log(websocket, header)

        # 打印本轮输入与文件摘要
        await _ws_log(websocket, f"> instruction.type={type(instruction).__name__}")
        try:
            preview = str(instruction)
            if len(preview) > 500:
                preview = preview[:500] + "... (truncated)"
            await _ws_log(websocket, f"> instruction.preview={preview}")
        except Exception:
            pass

        await _ws_log(websocket, _summarize_files(file_metadata))

        # 环境变量存在性（只显示是否设置）
        env_report = [
            "[env] MINIO_INTERNAL_ENDPOINT=" + _mask(os.getenv("MINIO_INTERNAL_ENDPOINT")),
            "[env] MINIO_PUBLIC_ENDPOINT=" + _mask(os.getenv("MINIO_PUBLIC_ENDPOINT")),
            "[env] MINIO_ACCESS_KEY_ID=" + _mask(os.getenv("MINIO_ACCESS_KEY_ID")),
            "[env] MINIO_ACCESS_KEY_SECRET=" + _mask(os.getenv("MINIO_ACCESS_KEY_SECRET")),
        ]
        await _ws_log(websocket, "\n".join(env_report))

        # 归一化文本
        debug_log: List[str] = []
        q = _norm_text(instruction, debug_log)
        await _ws_log(websocket, "\n".join(debug_log))
        await _ws_log(websocket, f"> 归一化后的当前用户问题：{q!r}")

        # 关键词（命中一个就停）
        trend_kws   = ["趋势", "trend", "trending", "slope", "移动平均", "滚动平均", "rolling mean", "moving average"]
        anomaly_kws = ["异常", "异常值", "异常点", "异动", "离群", "离群点", "异常检测", "异常分析",
                       "outlier", "outliers", "anomaly", "anomalies", "anomaly detection", "anomaly analysis"]
        corr_kws    = ["相关", "相关性", "自相关", "偏自相关", "acf", "pacf", "相关图"]
        fft_kws     = ["频谱", "频域", "fft", "周期", "周期性", "傅里叶", "fourier", "spectrum"]
        density_kws = ["密度", "密度图", "分布", "分布图", "kde", "density"]
        stats_kws   = ["统计", "统计信息", "summary", "describe", "info", "概览", "概述", "基本情况", "分析一下", "看一下"]

        routes: List[Tuple[str, str, List[str], bool, str, Dict[str, Any]]] = [
            ("team_trend",   "run_trend",   trend_kws,   True,  "趋势",          {}),
            ("team_anomaly", "run_anomaly", anomaly_kws, True,  "异常",          {}),
            ("team_stats",   "run_stats",   corr_kws,    True,  "相关(corr)",    {"submode": "corr"}),
            ("team_stats",   "run_stats",   fft_kws,     True,  "频谱(fft)",     {"submode": "fft"}),
            ("team_stats",   "run_stats",   density_kws, True,  "密度(density)", {"submode": "density"}),
            ("team_stats",   "run_stats",   stats_kws,   True,  "统计(stats)",   {"submode": "stats"}),
        ]

        # 遍历路由
        for module_name, func_name, kws, need_file, label, extra in routes:
            kw = _hit_any(q, kws)
            if not kw:
                continue

            await _ws_log(websocket, f"> 命中关键词：{kw!r} → 计划调用：{module_name}.{func_name}（{label}）")

            if need_file and not _has_files(file_metadata):
                await _ws_log(websocket, "> 触发分析模块但本轮未收到文件：前端目前每轮都要上传文件。")
                return ""

            # 动态导入
            try:
                run_fn, mod = _lazy_import(module_name, func_name)
                await _ws_log(websocket, f"> 已载入模块：{mod.__name__} | 源文件：{getattr(mod, '__file__', 'N/A')}")
                # 再确认函数签名存在
                if not callable(run_fn):
                    await _ws_log(websocket, f"> 警告：{module_name}.{func_name} 不是可调用对象")
            except Exception:
                tb = traceback.format_exc()
                await _ws_log(websocket, f"> 导入失败：{module_name}.{func_name}\n```\n{tb}\n```")
                # 导入失败才兜底聊天
                try:
                    run_chat, _mod = _lazy_import("team_chat", "run_chat")
                    return await run_chat(websocket, user_name, taskid, q)
                except Exception:
                    await _ws_log(websocket, f"> 兜底聊天导入也失败。\n```\n{traceback.format_exc()}\n```")
                    return ""

            # 真正执行
            try:
                t0 = time.time()
                await _ws_log(websocket, f"> 调用开始：{module_name}.{func_name} | args=submode={extra.get('submode')}")
                result = await run_fn(websocket, user_name, taskid, file_metadata, q, **(extra or {}))
                dt = time.time() - t0
                await _ws_log(websocket, f"> 调用完成：{module_name}.{func_name} | 耗时 {dt:.2f}s")
                return result if isinstance(result, str) else ""
            except Exception:
                tb = traceback.format_exc()
                await _ws_log(websocket, f"> 执行失败：{module_name}.{func_name}\n```\n{tb}\n```")
                # 执行失败再兜底聊天
                try:
                    run_chat, _mod = _lazy_import("team_chat", "run_chat")
                    return await run_chat(websocket, user_name, taskid, q)
                except Exception:
                    await _ws_log(websocket, f"> 兜底聊天执行也失败。\n```\n{traceback.format_exc()}\n```")
                    return ""

        # 未命中任何关键词 → 聊天
        await _ws_log(websocket, "> 未命中任何关键词：转通用对话（team_chat）")
        try:
            run_chat, _mod = _lazy_import("team_chat", "run_chat")
            return await run_chat(websocket, user_name, taskid, q)
        except Exception:
            await _ws_log(websocket, f"> 导入/执行 team_chat 失败。\n```\n{traceback.format_exc()}\n```")
            return ""


class DataRouter(Role):
    name: str = "路由器"
    profile: str = "根据提示词把请求分发给具体分析模块或聊天模块。"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([RouteAction])
