# src/team_router.py  (v7: force anomaly when files exist)
import os, re, json, time, types, importlib, traceback
from typing import Any, Dict, List, Optional, Tuple

from alpha.roles import Role
from alpha.actions import Action, UserRequirement

DEBUG = os.getenv("ROUTER_DEBUG", "1") != "0"

async def _ws_log(ws, msg: str):
    try:
        if DEBUG: print(msg, flush=True)
        await ws.send_text(msg + ("\n\n" if not msg.endswith("\n") else ""))
    except Exception:
        try:
            if DEBUG: print("[router-log-failed]", msg, flush=True)
        except Exception:
            pass

def _mask(s: Optional[str]) -> str:
    if not s: return "<unset>"
    return "***" + s[-4:] if len(s) > 4 else "***"

def _summarize_files(file_metadata: Any) -> str:
    if not isinstance(file_metadata, (list, tuple)) or not file_metadata:
        return "files: 0"
    lines = [f"files: {len(file_metadata)}"]
    keys = ("bucket","storage_key","key","path","original_filename","filename","name","file_name")
    for i, it in enumerate(file_metadata[:3], 1):
        if isinstance(it, dict):
            brief = {k: it.get(k) for k in keys if it.get(k) is not None}
            for k, v in list(brief.items()):
                if isinstance(v, str) and len(v) > 80: brief[k] = "..." + v[-77:]
            lines.append(f"  [{i}] {json.dumps(brief, ensure_ascii=False)}")
        else:
            s = str(it);  lines.append(f"  [{i}] {'...' + s[-77:] if len(s)>80 else s}")
    if len(file_metadata) > 3:
        lines.append(f"  ... (+{len(file_metadata)-3} more)")
    return "\n".join(lines)

def _lazy_import(module_name: str, func_name: str):
    tried = []
    def _try(mn: str):
        tried.append(mn)
        mod = importlib.import_module(mn)
        return getattr(mod, func_name), mod
    base = module_name if module_name.startswith("src.") else f"src.{module_name}"
    cands = [base]
    if base.endswith(".team_anomaly"): cands.append(base.replace("team_anomaly","team_anamoly"))
    if base.endswith(".team_anamoly"): cands.append(base.replace("team_anamoly","team_anomaly"))
    last = None
    for mn in cands:
        try:
            fn, mod = _try(mn);  return fn, mod
        except Exception as e: last = e
    raise ImportError(f"Cannot import {module_name}.{func_name}; tried={tried}; last_err={last}")

def _extract_text_role(obj: Any):
    if hasattr(obj, "content"):
        try:
            txt = getattr(obj, "content"); role = getattr(obj, "role", None)
            if isinstance(txt, str): return txt, (role if isinstance(role, str) else None)
        except Exception: pass
    if isinstance(obj, dict):
        for k in ("text","content","value","message"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip(): return v, (obj.get("role") if isinstance(obj.get("role"), str) else None)
    if isinstance(obj, str): return obj, None
    return "", None

def _norm_text(instruction: Any, debug: List[str]) -> str:
    cands: List[Tuple[str, Optional[str]]] = []
    if isinstance(instruction, (list, tuple)):
        for x in reversed(instruction):
            txt, role = _extract_text_role(x)
            if txt and txt.strip(): cands.append((txt, role))
    else:
        txt, role = _extract_text_role(instruction)
        if txt and txt.strip(): cands.append((txt, role))
    picked = ""
    for txt, role in cands:
        if role and str(role).lower() in {"user","human"}: picked = txt; break
    if not picked and cands: picked = cands[0][0]
    debug.append(f"[router] 候选条数: {len(cands)}")
    if cands:
        prev = picked[:200] + "..." if len(picked)>200 else picked
        debug.append(f"[router] 选用文本: {prev!r}")

    s = picked
    m = re.search(r"===\s*当前用户问题.*?===.*?(?:用户|user)\s*[:：]\s*(.+)$", s, flags=re.I|re.S)
    if m: s = m.group(1)
    s = re.sub(r"===\s*历史聊天记录.*?(?====|$)", " ", s, flags=re.I|re.S)
    s = re.sub(r"`{3,}.*?`{3,}", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
    s = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"^\s*(human|assistant|system)\s*:\s*", "", s, flags=re.I)
    for ch in "，。；：、！·（）【】《》“”": s = s.replace(ch, " ")
    s = re.sub(r"\s+"," ", s.replace("\u3000"," ")).strip().lower()
    return s

def _compile_kw(k: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z]){re.escape(k)}(?![a-z])", re.I) if re.search(r"[a-z]", k, re.I) else re.compile(re.escape(k))

def _hit_any(text: str, kws: List[str]) -> Optional[str]:
    for k in kws:
        if _compile_kw(k).search(text): return k
    return None

def _has_files(file_metadata: Any) -> bool:
    return isinstance(file_metadata, (list, tuple)) and len(file_metadata) > 0

class RouteAction(Action):
    name: str = "路由分析"
    desc: str = "根据提示词选择对应分析模块；未命中则走通用LLM对话"

    async def run(self, instruction: Any, *args):
        ws, user_name, taskid, file_metadata = args[0], args[1], args[2], args[3]

        # 明确打点：看到这三行就说明“确实进了路由器”
        await _ws_log(ws, "ROUTER_OK v7")
        await _ws_log(ws, f"> file={__file__}")
        await _ws_log(ws, f"> pid={os.getpid()} | cwd={os.getcwd()}")

        await _ws_log(ws, f"> instruction.type={type(instruction).__name__}")
        try:
            prev = str(instruction)
            if len(prev) > 500: prev = prev[:500] + "... (truncated)"
            await _ws_log(ws, f"> instruction.preview={prev}")
        except Exception: pass

        await _ws_log(ws, _summarize_files(file_metadata))

        # 1) 强制短路：只要这轮有文件，直接跑异常分析
        if _has_files(file_metadata):
            await _ws_log(ws, "> 触发短路策略：检测到本轮包含文件 → 直接调用 team_anomaly.run_anomaly")
            try:
                run_fn, mod = _lazy_import("team_anomaly", "run_anomaly")
                await _ws_log(ws, f"> 已载入模块：{mod.__name__} | 源文件：{getattr(mod,'__file__','N/A')}")
                t0 = time.time()
                res = await run_fn(ws, user_name, taskid, file_metadata, "anomaly (forced)")
                await _ws_log(ws, f"> 调用完成：team_anomaly.run_anomaly | 耗时 {time.time()-t0:.2f}s")
                return res if isinstance(res, str) else ""
            except Exception:
                await _ws_log(ws, f"> 执行失败：team_anomaly.run_anomaly\n```\n{traceback.format_exc()}\n```")
                # 失败再兜底聊天
                try:
                    run_chat, _ = _lazy_import("team_chat", "run_chat")
                    return await run_chat(ws, user_name, taskid, "anomaly fallback")
                except Exception:
                    await _ws_log(ws, f"> 兜底聊天也失败\n```\n{traceback.format_exc()}\n```")
                    return ""

        # 2) 没有文件时，再走关键词路由（原逻辑）
        debug: List[str] = []
        q = _norm_text(instruction, debug)
        await _ws_log(ws, "\n".join(debug))
        await _ws_log(ws, f"> 归一化后的当前用户问题：{q!r}")

        trend_kws   = ["趋势","trend","trending","slope","移动平均","滚动平均","rolling mean","moving average"]
        anomaly_kws = ["异常","异常值","异常点","异动","离群","离群点","异常检测","异常分析",
                       "outlier","outliers","anomaly","anomalies","anomaly detection","anomaly analysis"]
        corr_kws    = ["相关","相关性","自相关","偏自相关","acf","pacf","相关图"]
        fft_kws     = ["频谱","频域","fft","周期","周期性","傅里叶","fourier","spectrum"]
        density_kws = ["密度","密度图","分布","分布图","kde","density"]
        stats_kws   = ["统计","统计信息","summary","describe","info","概览","概述","基本情况","分析一下","看一下"]

        routes = [
            ("team_trend",   "run_trend",   trend_kws,   True,  "趋势",          {}),
            ("team_anomaly", "run_anomaly", anomaly_kws, True,  "异常",          {}),
            ("team_stats",   "run_stats",   corr_kws,    True,  "相关(corr)",    {"submode": "corr"}),
            ("team_stats",   "run_stats",   fft_kws,     True,  "频谱(fft)",     {"submode": "fft"}),
            ("team_stats",   "run_stats",   density_kws, True,  "密度(density)", {"submode": "density"}),
            ("team_stats",   "run_stats",   stats_kws,   True,  "统计(stats)",   {"submode": "stats"}),
        ]

        for module_name, func_name, kws, need_file, label, extra in routes:
            kw = _hit_any(q, kws)
            if not kw: continue
            await _ws_log(ws, f"> 命中关键词：{kw!r} → 调用：{module_name}.{func_name}（{label}）")
            try:
                run_fn, mod = _lazy_import(module_name, func_name)
                await _ws_log(ws, f"> 已载入：{mod.__name__} | 源：{getattr(mod,'__file__','N/A')}")
                t0 = time.time()
                res = await run_fn(ws, user_name, taskid, file_metadata, q, **(extra or {}))
                await _ws_log(ws, f"> 完成：{module_name}.{func_name} | 耗时 {time.time()-t0:.2f}s")
                return res if isinstance(res, str) else ""
            except Exception:
                await _ws_log(ws, f"> 执行失败：{module_name}.{func_name}\n```\n{traceback.format_exc()}\n```")
                try:
                    run_chat, _ = _lazy_import("team_chat", "run_chat")
                    return await run_chat(ws, user_name, taskid, q)
                except Exception:
                    await _ws_log(ws, f"> 兜底聊天失败\n```\n{traceback.format_exc()}\n```")
                    return ""

        await _ws_log(ws, "> 未命中任何关键词：转通用对话")
        try:
            run_chat, _ = _lazy_import("team_chat", "run_chat")
            return await run_chat(ws, user_name, taskid, q)
        except Exception:
            await _ws_log(ws, f"> 导入/执行 team_chat 失败\n```\n{traceback.format_exc()}\n```")
            return ""

class DataRouter(Role):
    name: str = "路由器"
    profile: str = "根据提示词把请求分发给具体分析模块或聊天模块。"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([RouteAction])
