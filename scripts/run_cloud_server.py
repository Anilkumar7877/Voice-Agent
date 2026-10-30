import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from orchestrator.config import config
from orchestrator.session.browser_session import BrowserSession

PORT = int(os.environ.get("PORT", 8765))
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_active = False  # one live session at a time: protects your API quota and RAM


async def handler(ws):
    global _active
    if _active:
        await ws.send(json.dumps({"type": "busy"}))
        await ws.close()
        return

    _active = True
    session = BrowserSession(ws, asyncio.get_running_loop())
    session.start()
    try:
        async for msg in ws:
            session.handle_message(msg)
    finally:
        session.stop()
        _active = False


async def main():
    config.validate()
    async with websockets.serve(handler, "0.0.0.0", PORT, origins=ORIGINS or None, max_size=2**20):
        print(f"[cloud] listening on :{PORT}, origins={ORIGINS or 'any'}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
