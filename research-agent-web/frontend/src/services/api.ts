// API service for Research Agent Web Application
import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ResearchRequest,
  ResearchResult,
  ResearchHistory,
  ExportRequest,
  HealthResponse,
  ResearchSettings,
  ErrorResponse
} from '../types/research';

class APIService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    // Use environment variable or default to localhost
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    this.api = axios.create({
      baseURL: `${this.baseURL}/api`,
      timeout: 900000, // 15 minutes for research operations (increased from 5 minutes)
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('🔴 API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('🔴 API Response Error:', error.response?.data || error.message);
        return Promise.reject(this.formatError(error));
      }
    );
  }

  private formatError(error: any): Error {
    if (error.response?.data) {
      const errorData = error.response.data as ErrorResponse;
      return new Error(errorData.message || errorData.error || 'API Error');
    }
    if (error.message) {
      return new Error(error.message);
    }
    return new Error('Unknown API Error');
  }

  // Health check
  async healthCheck(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await this.api.get('/health');
    return response.data;
  }

  // Research operations
  async conductResearch(request: ResearchRequest): Promise<ResearchResult> {
    const response: AxiosResponse<ResearchResult> = await this.api.post('/research/conduct', request);
    return response.data;
  }

  async getResearchResult(researchId: string): Promise<ResearchResult> {
    const response: AxiosResponse<ResearchResult> = await this.api.get(`/research/result/${researchId}`);
    return response.data;
  }

  // History operations
  async getResearchHistory(page: number = 1, pageSize: number = 10): Promise<ResearchHistory> {
    const response: AxiosResponse<ResearchHistory> = await this.api.get('/research/history', {
      params: { page, page_size: pageSize }
    });
    return response.data;
  }

  // Export operations
  async exportResearch(request: ExportRequest): Promise<Blob> {
    const response: AxiosResponse<Blob> = await this.api.post('/research/export', request, {
      responseType: 'blob'
    });
    return response.data;
  }

  // Session management
  async getActiveSessions(): Promise<{ active_sessions: Record<string, any>; total_active: number }> {
    const response = await this.api.get('/research/active-sessions');
    return response.data;
  }

  async cancelResearch(researchId: string): Promise<{ message: string }> {
    const response = await this.api.delete(`/research/session/${researchId}`);
    return response.data;
  }

  // Settings
  async getResearchSettings(): Promise<ResearchSettings & { enhanced_mode_available: boolean; openai_configured: boolean }> {
    const response = await this.api.get('/research/settings');
    return response.data;
  }

  // WebSocket URL helper
  getWebSocketURL(clientId: string): string {
    const wsProtocol = this.baseURL.startsWith('https://') ? 'wss:' : 'ws:';
    const wsBaseURL = this.baseURL.replace(/^https?:/, wsProtocol);
    return `${wsBaseURL}/api/ws/research/${clientId}`;
  }

  // Download helper for export
  downloadFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

// WebSocket service for real-time communication
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;  // Increased from 5
  private reconnectInterval = 1000;
  private clientId: string;
  private messageHandlers: Map<string, (data: any) => void> = new Map();
  private url: string = '';
  private isManuallyDisconnected = false;
  private heartbeatInterval: number | null = null;
  private lastPongReceived = Date.now();
  private isResearchActive = false;

  constructor(clientId: string) {
    this.clientId = clientId;
  }

  connect(url: string): Promise<void> {
    this.url = url;
    this.isManuallyDisconnected = false;
    
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('🔌 WebSocket connected');
          this.reconnectAttempts = 0;
          this.lastPongReceived = Date.now();
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            console.log('📨 WebSocket message:', message);
            
            // Handle pong messages for heartbeat
            if (message.type === 'pong') {
              this.lastPongReceived = Date.now();
              return;
            }
            
            const handler = this.messageHandlers.get(message.type);
            if (handler) {
              handler(message.data);
            }
          } catch (error) {
            console.error('🔴 WebSocket message parse error:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('🔌 WebSocket disconnected:', event.code, event.reason);
          this.stopHeartbeat();
          
          // If we were in the middle of research, signal completion to prevent hanging
          if (this.isResearchActive) {
            console.log('🔬 Research was active during disconnect - signaling completion');
            this.setResearchComplete();
            
            // Trigger error handler to show user-friendly message
            const errorHandler = this.messageHandlers.get('error');
            if (errorHandler) {
              errorHandler({
                message: 'Connection lost during research. Results may be incomplete. Please try again.'
              });
            }
          }
          
          if (!this.isManuallyDisconnected) {
            this.handleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('🔴 WebSocket error:', error);
          this.stopHeartbeat();
          
          // Don't reject immediately - let the reconnect logic handle it
          if (this.reconnectAttempts === 0) {
            reject(error);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private handleReconnect(): void {
    if (this.isManuallyDisconnected) {
      return;
    }
    
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1), 30000); // Exponential backoff, max 30s
      console.log(`🔄 Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
      
      setTimeout(() => {
        if (!this.isManuallyDisconnected) {
          this.connect(this.url).catch((error) => {
            console.error('🔴 Reconnection failed:', error);
          });
        }
      }, delay);
    } else {
      console.error('🔴 Max WebSocket reconnection attempts reached');
      // Trigger a connection error event that the UI can handle
      const handler = this.messageHandlers.get('connection_failed');
      if (handler) {
        handler({ message: 'Connection failed after maximum retry attempts' });
      }
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat(); // Clear any existing heartbeat
    
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        // Skip heartbeat during active research to prevent interference
        if (this.isResearchActive) {
          console.log('🔬 Skipping heartbeat during active research');
          this.lastPongReceived = Date.now(); // Reset to prevent timeout
          return;
        }
        
        // Check if we received a pong recently (increased timeout for research operations)
        const timeSinceLastPong = Date.now() - this.lastPongReceived;
        const timeoutDuration = this.isResearchActive ? 300000 : 120000; // 5 minutes during research, 2 minutes normally
        if (timeSinceLastPong > timeoutDuration) {
          console.warn('⚠️ WebSocket heartbeat timeout, forcing reconnection');
          this.ws.close();
          return;
        }
        
        // Send ping
        this.ping();
      }
    }, 30000); // Send ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  onMessage(type: string, handler: (data: any) => void): void {
    this.messageHandlers.set(type, handler);
  }

  sendMessage(type: string, data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message = {
        type,
        data,
        timestamp: new Date().toISOString()
      };
      this.ws.send(JSON.stringify(message));
      console.log('📤 WebSocket message sent:', message);
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send message');
    }
  }

  startResearch(request: ResearchRequest): void {
    this.isResearchActive = true;
    console.log('🚀 Starting research - disabling heartbeat');
    this.sendMessage('start_research', request);
  }

  setResearchComplete(): void {
    this.isResearchActive = false;
    console.log('✅ Research complete - re-enabling heartbeat');
    this.lastPongReceived = Date.now(); // Reset heartbeat timer
  }

  subscribe(researchId: string): void {
    this.sendMessage('subscribe', { research_id: researchId });
  }

  unsubscribe(researchId: string): void {
    this.sendMessage('unsubscribe', { research_id: researchId });
  }

  ping(): void {
    this.sendMessage('ping', {});
  }

  disconnect(): void {
    this.isManuallyDisconnected = true;
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const apiService = new APIService();
export default apiService;