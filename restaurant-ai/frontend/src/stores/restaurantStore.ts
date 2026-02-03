import { create } from 'zustand';
import type { Table, Staff, Event, Notification, TableState } from '../types';

interface RestaurantState {
  // Data
  tables: Table[];
  staff: Staff[];
  events: Event[];
  notifications: Notification[];

  // Connection
  connectionStatus: 'connected' | 'disconnected' | 'error';

  // UI state
  selectedTableId: number | null;
  selectedStaffId: number | null;
  showNotifications: boolean;

  // Actions
  setTables: (tables: Table[]) => void;
  setStaff: (staff: Staff[]) => void;
  setEvents: (events: Event[]) => void;
  addEvent: (event: Event) => void;
  updateTable: (id: number, updates: Partial<Table>) => void;
  updateStaff: (id: number, updates: Partial<Staff>) => void;
  addNotification: (notification: Notification) => void;
  dismissNotification: (id: string) => void;
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'error') => void;
  selectTable: (id: number | null) => void;
  selectStaff: (id: number | null) => void;
  toggleNotifications: () => void;
}

export const useRestaurantStore = create<RestaurantState>((set) => ({
  // Initial state
  tables: [],
  staff: [],
  events: [],
  notifications: [],
  connectionStatus: 'disconnected',
  selectedTableId: null,
  selectedStaffId: null,
  showNotifications: false,

  // Actions
  setTables: (tables) => set({ tables }),

  setStaff: (staff) => set({ staff }),

  setEvents: (events) => set({ events }),

  addEvent: (event) =>
    set((state) => ({
      events: [event, ...state.events].slice(0, 100), // Keep last 100 events
    })),

  updateTable: (id, updates) =>
    set((state) => ({
      tables: state.tables.map((table) =>
        table.id === id ? { ...table, ...updates } : table
      ),
    })),

  updateStaff: (id, updates) =>
    set((state) => ({
      staff: state.staff.map((s) =>
        s.id === id ? { ...s, ...updates } : s
      ),
    })),

  addNotification: (notification) =>
    set((state) => ({
      notifications: [notification, ...state.notifications].slice(0, 50),
      showNotifications: true,
    })),

  dismissNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  selectTable: (id) => set({ selectedTableId: id }),

  selectStaff: (id) => set({ selectedStaffId: id }),

  toggleNotifications: () =>
    set((state) => ({ showNotifications: !state.showNotifications })),
}));

// Selectors
export const selectTablesByState = (state: RestaurantState, tableState: TableState) =>
  state.tables.filter((t) => t.state === tableState);

export const selectAvailableStaff = (state: RestaurantState) =>
  state.staff.filter((s) => s.status === 'available');

export const selectUnresolvedEvents = (state: RestaurantState) =>
  state.events.filter((e) => !e.resolved);

export const selectTableStats = (state: RestaurantState) => {
  const stats: Record<TableState, number> = {
    empty: 0,
    seated: 0,
    ordering: 0,
    waiting: 0,
    served: 0,
    paying: 0,
  };

  state.tables.forEach((t) => {
    stats[t.state]++;
  });

  return stats;
};

export const selectTotalGuests = (state: RestaurantState) =>
  state.tables.reduce((sum, t) => sum + t.current_guests, 0);
