import websockets
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_heartbeat: Dict[str, datetime] = {}
        self.HEARTBEAT_TIMEOUT = 60  # 60 seconds

    async def connect(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Add a new WebSocket connection"""
        self.connections[client_id] = websocket
        self._last_heartbeat[client_id] = datetime.now()
        logger.info(f"Client {client_id} connected")

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.connections:
            try:
                websocket = self.connections[client_id]
                if not websocket.client_state.DISCONNECTED:
                    await websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection for client {client_id}: {str(e)}")
            finally:
                del self.connections[client_id]
                if client_id in self._last_heartbeat:
                    del self._last_heartbeat[client_id]
                logger.info(f"Client {client_id} disconnected")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.connections:
            return

        message_json = json.dumps(message)
        disconnected_clients = []

        for client_id, websocket in self.connections.items():
            try:
                if not websocket.client_state.DISCONNECTED:
                    await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client {client_id} connection closed")
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

    def start(self):
        """Start the WebSocket manager"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())

    def stop(self):
        """Stop the WebSocket manager"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _check_heartbeats(self):
        """Check for stale connections and clean them up"""
        now = datetime.now()
        stale_clients = [
            client_id for client_id, last_heartbeat in self._last_heartbeat.items()
            if (now - last_heartbeat).total_seconds() > self.HEARTBEAT_TIMEOUT
        ]
        
        for client_id in stale_clients:
            logger.warning(f"Client {client_id} heartbeat timeout")
            await self.disconnect(client_id)

    async def _run(self):
        """Main loop for the WebSocket manager"""
        while self._running:
            try:
                await self._check_heartbeats()
                await asyncio.sleep(1)  # Prevent CPU overuse
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket manager loop: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

    async def handle_heartbeat(self, client_id: str):
        """Handle a heartbeat from a client"""
        if client_id in self._last_heartbeat:
            self._last_heartbeat[client_id] = datetime.now()
            return True
        return False 