'use client';

import { useEffect, useRef, useState } from 'react';
import { SessionProgressEvent } from '../lib/types';

export function useSessionWebSocket(sessionId: string | null) {
  const [progress, setProgress] = useState<SessionProgressEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setProgress(null);
      setIsConnected(false);
      return;
    }

    const wsUrl = `ws://127.0.0.1:8000/api/v1/sessions/${sessionId}/events`;
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data: SessionProgressEvent = JSON.parse(event.data);
        setProgress(data);
      } catch (err) {
        console.error('Failed to parse WebSocket progress event:', err);
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
    };

    socket.onerror = () => {
      setIsConnected(false);
    };

    return () => {
      socket.close();
    };
  }, [sessionId]);

  return { progress, isConnected };
}
