// Table types
export type TableState = 'empty' | 'seated' | 'ordering' | 'waiting' | 'served' | 'paying';

export interface Table {
  id: number;
  number: number;
  capacity: number;
  x_position: number;
  y_position: number;
  state: TableState;
  current_guests: number;
  state_changed_at: string | null;
  seated_at: string | null;
  assigned_server_id: number | null;
  priority_score: number;
  hand_raise_detected: boolean;
  last_interaction: string | null;
  zone: string;
  notes: string | null;
}

// Staff types
export type StaffRole = 'server' | 'busser' | 'host' | 'manager' | 'bartender';
export type StaffStatus = 'available' | 'busy' | 'on_break' | 'off_duty';

export interface Staff {
  id: number;
  name: string;
  role: StaffRole;
  status: StaffStatus;
  x_position: number;
  y_position: number;
  current_zone: string;
  assigned_tables: number[];
  current_task_count: number;
  max_tables: number;
  tasks_completed_today: number;
  avg_response_time: number;
  skills: string[];
  shift_start: string | null;
  shift_end: string | null;
  last_activity: string | null;
}

// Event types
export type EventType =
  | 'customer_seated'
  | 'customer_left'
  | 'hand_raise'
  | 'menu_request'
  | 'order_placed'
  | 'food_delivered'
  | 'check_request'
  | 'payment_complete'
  | 'staff_assigned'
  | 'staff_arrived'
  | 'task_completed'
  | 'verbal_request'
  | 'complaint'
  | 'priority_alert'
  | 'zone_change'
  | 'gesture_detected'
  | 'person_detected';

export type EventPriority = 'low' | 'medium' | 'high' | 'urgent';

export interface Event {
  id: number;
  event_type: EventType;
  priority: EventPriority;
  timestamp: string;
  table_id: number | null;
  staff_id: number | null;
  zone: string | null;
  x_position: number | null;
  y_position: number | null;
  confidence: number;
  source: string;
  description: string;
  metadata: Record<string, unknown>;
  resolved: boolean;
  resolved_at: string | null;
  resolved_by: number | null;
}

// Notification types
export interface Notification {
  id: string;
  title: string;
  message: string;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  channels: string[];
  target_staff_ids: number[];
  table_id: number | null;
  event_id: number | null;
  created_at: string;
  delivered: boolean;
  acknowledged: boolean;
}

// WebSocket message types
export interface WSMessage {
  type: string;
  data: unknown;
  timestamp?: string;
}

export interface InitialState {
  tables: Table[];
  staff: Staff[];
  events: Event[];
}

export interface StateUpdate {
  tables: Record<number, {
    state: TableState;
    time_in_state: number;
    is_overdue: boolean;
    person_count: number;
    assigned_staff_id: number | null;
  }>;
  queue_stats: QueueStats;
  restaurant_stats: RestaurantStats;
}

// Statistics types
export interface QueueStats {
  total_pending: number;
  total_assigned: number;
  avg_wait_seconds: number;
  max_wait_seconds: number;
  by_type: Record<string, number>;
}

export interface RestaurantStats {
  total_tables: number;
  active_tables: number;
  overdue_tables: number;
  state_counts: Record<TableState, number>;
  total_guests: number;
}

// Analytics types
export interface AnalyticsSummary {
  time_range_minutes: number;
  wait_times: {
    count: number;
    avg: number;
    min: number;
    max: number;
  };
  response_times: {
    count: number;
    avg: number;
    min: number;
    max: number;
  };
  events: Record<string, number>;
  tables_active: number;
  staff_active: number;
}

// Zone types
export interface Zone {
  id: string;
  name: string;
  zone_type: string;
  bounds: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
  center: [number, number];
  capacity: number;
  current_occupancy: number;
  occupancy_ratio: number;
  is_active: boolean;
  tables: number[];
}
