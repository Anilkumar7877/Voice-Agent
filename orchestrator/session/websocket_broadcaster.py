import asyncio
import json
import threading
import websockets

class WebSocketBroadcaster:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None
        self._lock = threading.Lock()
        self.on_clear_history = None

    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        with self._lock:
            if self.thread is not None:
                return  # Already running
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="WebSocketBroadcasterThread"
            )
            self.thread.start()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())
        self.loop.run_forever()

    async def _start_server(self):
        try:
            # Bind the handler
            self.server = await websockets.serve(self._handler, self.host, self.port)
            print(f"[WebSocket] Server listening on ws://{self.host}:{self.port}")
        except Exception as e:
            print(f"[WebSocket] Failed to start server: {e}")

    async def _handler(self, websocket, path=None):
        with self._lock:
            self.clients.add(websocket)
        print(f"[WebSocket] Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "clear_history" and self.on_clear_history:
                        self.on_clear_history()
                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._lock:
                self.clients.discard(websocket)
            print(f"[WebSocket] Client disconnected: {websocket.remote_address}")

    async def _send_to_clients(self, event):
        with self._lock:
            if not self.clients:
                return
            clients_copy = list(self.clients)
        
        message = json.dumps(event)
        if clients_copy:
            await asyncio.gather(
                *[client.send(message) for client in clients_copy],
                return_exceptions=True
            )

    def broadcast(self, event):
        """Thread-safe method to broadcast an event to all connected clients."""
        if self.loop and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._send_to_clients(event), self.loop)
            except Exception:
                pass  # Ignore exceptions during thread teardown

# Global singleton instance
broadcaster = WebSocketBroadcaster()
