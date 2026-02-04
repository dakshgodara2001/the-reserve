import React from 'react';
import { useRestaurantStore } from '../stores/restaurantStore';
import type { Event, EventType, EventPriority } from '../types';

const PRIORITY_COLORS: Record<EventPriority, string> = {
  low: 'bg-gray-100 text-gray-800 border-gray-300',
  medium: 'bg-blue-100 text-blue-800 border-blue-300',
  high: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  urgent: 'bg-red-100 text-red-800 border-red-300',
};

const EVENT_ICONS: Partial<Record<EventType, string>> = {
  hand_raise: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
  customer_seated: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z',
  order_placed: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01',
  food_delivered: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
  check_request: 'M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z',
  complaint: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  staff_assigned: 'M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z',
};

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) {
    return 'Just now';
  } else if (diff < 3600000) {
    const mins = Math.floor(diff / 60000);
    return `${mins}m ago`;
  } else {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
}

function getEventDescription(event: Event): string {
  const tableInfo = event.table_id ? `Table ${event.table_id}` : '';

  switch (event.event_type) {
    case 'customer_seated':
      return `${tableInfo}: Customers seated`;
    case 'hand_raise':
      return `${tableInfo}: Hand raise detected`;
    case 'order_placed':
      return `${tableInfo}: Order placed`;
    case 'food_delivered':
      return `${tableInfo}: Food delivered`;
    case 'check_request':
      return `${tableInfo}: Check requested`;
    case 'complaint':
      return `${tableInfo}: Complaint received`;
    case 'staff_assigned':
      return `Staff assigned to ${tableInfo}`;
    case 'verbal_request':
      return `${tableInfo}: Verbal request`;
    default:
      return event.description || event.event_type.replace(/_/g, ' ');
  }
}

interface EventItemProps {
  event: Event;
}

function EventItem({ event }: EventItemProps) {
  const iconPath = EVENT_ICONS[event.event_type] || 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z';

  return (
    <div
      className={`p-3 rounded-lg border-l-4 ${PRIORITY_COLORS[event.priority]} ${
        event.resolved ? 'opacity-50' : ''
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="flex-shrink-0">
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d={iconPath}
            />
          </svg>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">{getEventDescription(event)}</p>
          {event.description && event.description !== getEventDescription(event) && (
            <p className="text-xs text-gray-500 mt-1 truncate">{event.description}</p>
          )}
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-gray-500">{formatTimestamp(event.timestamp)}</span>
            <span className="text-xs text-gray-400">|</span>
            <span className="text-xs text-gray-500 capitalize">{event.source}</span>
            {event.confidence < 1 && (
              <>
                <span className="text-xs text-gray-400">|</span>
                <span className="text-xs text-gray-500">
                  {Math.round(event.confidence * 100)}% conf
                </span>
              </>
            )}
          </div>
        </div>

        {/* Resolved badge */}
        {event.resolved && (
          <span className="flex-shrink-0 px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded">
            Resolved
          </span>
        )}
      </div>
    </div>
  );
}

export function EventFeed() {
  const events = useRestaurantStore((state) => state.events);
  const [filter, setFilter] = React.useState<'all' | 'unresolved' | EventPriority>('all');

  const filteredEvents = React.useMemo(() => {
    let filtered = [...events];

    if (filter === 'unresolved') {
      filtered = filtered.filter((e) => !e.resolved);
    } else if (filter !== 'all') {
      filtered = filtered.filter((e) => e.priority === filter);
    }

    return filtered.slice(0, 50);
  }, [events, filter]);

  const unresolvedCount = events.filter((e) => !e.resolved).length;
  const urgentCount = events.filter((e) => e.priority === 'urgent' && !e.resolved).length;

  return (
    <div className="bg-white rounded-lg shadow p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">Events</h2>
          {urgentCount > 0 && (
            <span className="px-2 py-0.5 text-xs bg-red-100 text-red-700 rounded-full animate-pulse">
              {urgentCount} urgent
            </span>
          )}
        </div>
        <span className="text-sm text-gray-500">{unresolvedCount} pending</span>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {(['all', 'unresolved', 'urgent', 'high', 'medium', 'low'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === f
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Event list */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredEvents.length === 0 ? (
          <div className="text-center text-gray-500 py-8">No events to display</div>
        ) : (
          filteredEvents.map((event) => <EventItem key={event.id} event={event} />)
        )}
      </div>
    </div>
  );
}
