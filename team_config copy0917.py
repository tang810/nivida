# analysis_data/team_config.py —— 单功能：统计信息（stats）
from typing import Any, Optional, Tuple
from dotenv import load_dotenv
from alpha.roles import Role
from alpha.actions import Action
from src.team_stats import run_stats  # 复用你已有的统计函数

load_dotenv()
SERVICE_NAME = "stats"

# —— 新增：把“分析一下/看看数据”等模糊请求当作统计的默认触发词
GENERIC_TRIGGERS = [
    "分析一下", "分析下", "帮我分析", "看看数据", "看下数据", "看看情况",
    "看一下", "给我看看", "简单看下", "解读一下",
    "analysis", "analyze", "overview", "summary",
]

def _extract_last_user_text(history: Any) -> str:
    """从 Alpha 的 history 中尽量取出最近一条用户消息文本。"""
    if not history:
        return ""
    def _get(obj: Any) -> Tuple[str, Optional[str]]:
        if hasattr(obj, "content"):
            try:
                t = getattr(obj, "content")
                r = getattr(obj, "role", None)
                if isinstance(t, str):
                    return t, (r if isinstance(r, str) else None)
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

    try:
        for x in reversed(history):
            t, r = _get(x)
            if t.strip() and r and str(r).lower() in {"user", "human"}:
                return t
        for x in reversed(history):
            t, _ = _get(x)
            if t.strip():
                return t
    except Exception:
        pass
    return ""

def _is_generic_analysis(q: str) -> bool:
    """弱意图：无“统计/趋势/异常”等明确词，但看起来是泛分析请求。"""
    q = (q or "").strip()
    if not q:
        return True
    ql = q.lower()
    return any(tok in q or tok in ql for tok in GENERIC_TRIGGERS)

class StatsAction(Action):
    name: str = "执行统计信息"
    desc: str = "调用 src.team_stats.run_stats(submode='stats') 输出数据概览（默认兜底）"

    async def run(self, history, websocket, user_name, taskid, file_metadata):
        # 注意：main.py 会把 history 作为第一个参数传进来，这里兼容并自行取出用户问题
        q = _extract_last_user_text(history) or ""
        # 泛分析→给统计模块一个“更全面”的计划；明确需求→走轻量
        plan = "full" if _is_generic_analysis(q) else "quick"

        # 可选：把路由决定告诉前端（便于调试）
        await websocket.send_text(f"[{SERVICE_NAME}] plan={plan}")

        return await run_stats(
            websocket=websocket,
            user_name=user_name,
            taskid=taskid,
            file_metadata=file_metadata,
            q=q,
            submode="stats",   # 走统计分支
            plan=plan,         # 关键：把 plan 传给统计实现
        )

class data_analysis_Input_Analyst(Role):
    name: str = "统计信息"
    profile: str = "单功能服务：统计信息（summary/describe/info 等，默认兜底）"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([StatsAction()])

__all__ = ["data_analysis_Input_Analyst", "load_dotenv"]
