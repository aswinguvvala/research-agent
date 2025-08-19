"""
WebSocket API endpoints for real-time research progress
"""

import asyncio
import json
import logging
from typing import Dict, Set
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState

from ..models.research import (
    ResearchRequest, ResearchProgress, WSMessage, WSResearchUpdate, WSError
)
from ..services.research_service import research_service
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.research_subscriptions: Dict[str, Set[str]] = {}  # research_id -> set of client_ids
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"🔌 WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from research subscriptions
        for research_id, subscribers in self.research_subscriptions.items():
            if client_id in subscribers:
                subscribers.remove(client_id)
        
        logger.info(f"🔌 WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(json.dumps(message, default=str))
                except Exception as e:
                    logger.error(f"❌ Failed to send message to {client_id}: {e}")
                    self.disconnect(client_id)
    
    async def broadcast_research_update(self, research_id: str, progress: ResearchProgress):
        """Broadcast research progress to all subscribed clients."""
        if research_id in self.research_subscriptions:
            message = WSResearchUpdate(
                research_id=research_id,
                progress=progress
            )
            
            ws_message = WSMessage(
                type="research_progress",
                data=message.dict(),
                timestamp=datetime.now()
            )
            
            # Send to all subscribed clients
            subscribers = list(self.research_subscriptions[research_id])
            for client_id in subscribers:
                await self.send_personal_message(ws_message.dict(), client_id)
    
    def subscribe_to_research(self, client_id: str, research_id: str):
        """Subscribe client to research updates."""
        if research_id not in self.research_subscriptions:
            self.research_subscriptions[research_id] = set()
        
        self.research_subscriptions[research_id].add(client_id)
        logger.info(f"📡 Client {client_id} subscribed to research {research_id}")
    
    def unsubscribe_from_research(self, client_id: str, research_id: str):
        """Unsubscribe client from research updates."""
        if research_id in self.research_subscriptions:
            self.research_subscriptions[research_id].discard(client_id)
            
            # Clean up empty subscriptions
            if not self.research_subscriptions[research_id]:
                del self.research_subscriptions[research_id]
        
        logger.info(f"📡 Client {client_id} unsubscribed from research {research_id}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "active_research_subscriptions": len(self.research_subscriptions),
            "total_subscribers": sum(len(subs) for subs in self.research_subscriptions.values())
        }

# Global connection manager
manager = ConnectionManager()

@router.websocket("/research/{client_id}")
async def research_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time research communication.
    
    Messages format:
    - Client -> Server: {"type": "start_research", "data": ResearchRequest}
    - Client -> Server: {"type": "subscribe", "data": {"research_id": "..."}}
    - Server -> Client: {"type": "research_progress", "data": WSResearchUpdate}
    - Server -> Client: {"type": "research_complete", "data": ResearchResult}
    - Server -> Client: {"type": "error", "data": WSError}
    """
    
    # Validate client ID
    if not client_id or len(client_id) < 3:
        await websocket.close(code=4000, reason="Invalid client ID")
        return
    
    # Check connection limits
    if len(manager.active_connections) >= settings.WS_MAX_CONNECTIONS:
        await websocket.close(code=4000, reason="Connection limit reached")
        return
    
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        welcome_msg = WSMessage(
            type="connected",
            data={"client_id": client_id, "message": "Connected to Research Agent"},
            timestamp=datetime.now()
        )
        await manager.send_personal_message(welcome_msg.dict(), client_id)
        
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Validate message format
                if "type" not in message or "data" not in message:
                    await _send_error(client_id, "Invalid message format")
                    continue
                
                message_type = message["type"]
                message_data = message["data"]
                
                if message_type == "start_research":
                    await _handle_start_research(client_id, message_data)
                elif message_type == "subscribe":
                    await _handle_subscribe(client_id, message_data)
                elif message_type == "unsubscribe":
                    await _handle_unsubscribe(client_id, message_data)
                elif message_type == "ping":
                    await _handle_ping(client_id)
                else:
                    await _send_error(client_id, f"Unknown message type: {message_type}")
                
            except json.JSONDecodeError:
                await _send_error(client_id, "Invalid JSON format")
            except Exception as e:
                logger.error(f"❌ WebSocket message handling error: {e}")
                await _send_error(client_id, f"Message handling error: {str(e)}")
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected normally: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

async def _handle_start_research(client_id: str, data: dict):
    """Handle start research request."""
    try:
        # Parse research request
        research_request = ResearchRequest(**data)
        
        logger.info(f"🔍 WebSocket research request from {client_id}: {research_request.query}")
        
        # Create progress callback
        async def progress_callback(progress: ResearchProgress):
            await manager.send_personal_message({
                "type": "research_progress",
                "data": {
                    "research_id": "current",  # Will be updated with actual ID
                    "progress": progress.dict()
                },
                "timestamp": datetime.now().isoformat()
            }, client_id)
        
        # Start research (async in background)
        asyncio.create_task(_conduct_research_with_updates(client_id, research_request, progress_callback))
        
        # Send acknowledgment
        await manager.send_personal_message({
            "type": "research_started",
            "data": {"message": "Research started"},
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
    except Exception as e:
        logger.error(f"❌ Start research error: {e}")
        await _send_error(client_id, f"Failed to start research: {str(e)}")

async def _conduct_research_with_updates(client_id: str, request: ResearchRequest, progress_callback):
    """Conduct research and send updates via WebSocket."""
    try:
        # Conduct research with progress updates
        result = await research_service.conduct_research(request, progress_callback)
        
        # Send completion message
        await manager.send_personal_message({
            "type": "research_complete",
            "data": result.dict(),
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
        logger.info(f"✅ WebSocket research completed for {client_id}: {result.research_id}")
        
    except Exception as e:
        logger.error(f"❌ WebSocket research error for {client_id}: {e}")
        await _send_error(client_id, f"Research failed: {str(e)}")

async def _handle_subscribe(client_id: str, data: dict):
    """Handle subscription to research updates."""
    try:
        research_id = data.get("research_id")
        if not research_id:
            await _send_error(client_id, "Missing research_id")
            return
        
        manager.subscribe_to_research(client_id, research_id)
        
        await manager.send_personal_message({
            "type": "subscribed",
            "data": {"research_id": research_id},
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
    except Exception as e:
        await _send_error(client_id, f"Subscription error: {str(e)}")

async def _handle_unsubscribe(client_id: str, data: dict):
    """Handle unsubscription from research updates."""
    try:
        research_id = data.get("research_id")
        if not research_id:
            await _send_error(client_id, "Missing research_id")
            return
        
        manager.unsubscribe_from_research(client_id, research_id)
        
        await manager.send_personal_message({
            "type": "unsubscribed",
            "data": {"research_id": research_id},
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
    except Exception as e:
        await _send_error(client_id, f"Unsubscription error: {str(e)}")

async def _handle_ping(client_id: str):
    """Handle ping message."""
    await manager.send_personal_message({
        "type": "pong",
        "data": {"timestamp": datetime.now().isoformat()},
        "timestamp": datetime.now().isoformat()
    }, client_id)

async def _send_error(client_id: str, error_message: str):
    """Send error message to client."""
    error_msg = WSError(
        error="websocket_error",
        message=error_message,
        timestamp=datetime.now()
    )
    
    ws_message = WSMessage(
        type="error",
        data=error_msg.dict(),
        timestamp=datetime.now()
    )
    
    await manager.send_personal_message(ws_message.dict(), client_id)

@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return manager.get_stats()