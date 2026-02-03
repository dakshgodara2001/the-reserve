import { useEffect, useRef, useCallback, useState } from 'react';
import { useRestaurantStore } from '../stores/restaurantStore';
import type { WSMessage, InitialState, Table, Staff, Event, Notification, StateUpdate } from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const {
    setTables,
    setStaff,
    setEvents,
    addEvent,
    updateTable,
    updateStaff,
    addNotification,
    setConnectionStatus,
  } = useRestaurantStore();

  const handleMessage = useCallback((message: WSMessage) => {
    switch (message.type) {
      case 'initial_state': {
        const data = message.data as InitialState;
        setTables(data.tables);
        setStaff(data.staff);
        setEvents(data.events);
        break;
      }

      case 'state_update': {
        const data = message.data as StateUpdate;
        // Update individual table states from the update
        Object.entries(data.tables).forEach(([tableId, tableState]) => {
          updateTable(parseInt(tableId), {
            state: tableState.state,
            current_guests: tableState.person_count,
          });
        });
        break;
      }

      case 'table_created':
      case 'table_updated':
      case 'customers_seated': {
        const table = message.data as Table;
        updateTable(table.id, table);
        break;
      }

      case 'staff_created':
      case 'staff_updated': {
        const staff = message.data as Staff;
        updateStaff(staff.id, staff);
        break;
      }

      case 'table_assigned': {
        const data = message.data as { staff: Staff; table: Table };
        updateStaff(data.staff.id, data.staff);
        updateTable(data.table.id, data.table);
        break;
      }

      case 'new_event':
      case 'event_resolved': {
        const event = message.data as Event;
        addEvent(event);
        break;
      }

      case 'notification': {
        const notification = message.data as Notification;
        addNotification(notification);
        break;
      }

      case 'fused_event': {
        // Handle fused sensor events
        console.log('Fused event:', message.data);
        break;
      }

      case 'pong': {
        // Heartbeat response
        break;
      }

      default:
        console.log('Unknown message type:', message.type);
    }
  }, [setTables, setStaff, setEvents, addEvent, updateTable, updateStaff, addNotification]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        setConnectionStatus('connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WSMessage;
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setConnectionStatus('disconnected');

        // Attempt reconnection
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, RECONNECT_DELAY);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('Connection failed');
        setConnectionStatus('error');
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionError('Failed to connect');
    }
  }, [handleMessage, setConnectionStatus]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((type: string, data?: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, data }));
    }
  }, []);

  const requestState = useCallback(() => {
    sendMessage('request_state');
  }, [sendMessage]);

  const simulateEvent = useCallback((eventData: Record<string, unknown>) => {
    sendMessage('simulate_event', eventData);
  }, [sendMessage]);

  // Start heartbeat
  useEffect(() => {
    const heartbeatInterval = setInterval(() => {
      if (isConnected) {
        sendMessage('ping');
      }
    }, 30000);

    return () => clearInterval(heartbeatInterval);
  }, [isConnected, sendMessage]);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    isConnected,
    connectionError,
    sendMessage,
    requestState,
    simulateEvent,
    reconnect: connect,
  };
}
