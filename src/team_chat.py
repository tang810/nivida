# team_chat.py
from src.llm_utils import SeLLM, load_config

async def run_chat(websocket, user: str, taskid: str, instruction: str):
    await websocket.send_text("## 对话模式\n\n")
    cfg = load_config("config/config.yaml")
    llm = SeLLM(base_url=cfg["base_url_1"], api_key=cfg["api_key"])

    sys_msg = llm._default_system_msg()
    user_msg = llm._user_msg(instruction)
    stream = await llm.acompletion_text([sys_msg, user_msg], temperature=0.6, timeout=60)
    async for c in stream:
        await websocket.send_text(c.choices[0].delta.content or "" if c.choices else "")
    await websocket.send_text("\n")
    return ""
#hdue
