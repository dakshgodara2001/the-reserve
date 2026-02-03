import React from 'react';
import { FloorPlan } from './FloorPlan';
import { StaffList } from './StaffList';
import { EventFeed } from './EventFeed';
import { Analytics } from './Analytics';
import { useRestaurantStore } from '../stores/restaurantStore';
import { useWebSocket } from '../hooks/useWebSocket';

type TabType = 'overview' | 'analytics';

export function Dashboard() {
  const [activeTab, setActiveTab] = React.useState<TabType>('overview');
  const { isConnected, connectionError, simulateEvent } = useWebSocket();
  const connectionStatus = useRestaurantStore((state) => state.connectionStatus);
  const notifications = useRestaurantStore((state) => state.notifications);
  const showNotifications = useRestaurantStore((state) => state.showNotifications);
  const toggleNotifications = useRestaurantStore((state) => state.toggleNotifications);
  const dismissNotification = useRestaurantStore((state) => state.dismissNotification);

  // Simulate events for testing
  const handleSimulateHandRaise = () => {
    simulateEvent({
      event_type: 'hand_raise',
      table_id: Math.floor(Math.random() * 10) + 1,
      priority: 'high',
      description: 'Simulated hand raise',
    });
  };

  const handleSimulateSeating = () => {
    simulateEvent({
      event_type: 'customer_seated',
      table_id: Math.floor(Math.random() * 10) + 1,
      priority: 'medium',
      description: 'Simulated customer seating',
    });
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold text-gray-900">Restaurant AI Dashboard</h1>

              {/* Connection status */}
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected'
                      ? 'bg-green-500'
                      : connectionStatus === 'error'
                      ? 'bg-red-500'
                      : 'bg-yellow-500'
                  }`}
                />
                <span className="text-sm text-gray-500 capitalize">{connectionStatus}</span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Simulation buttons (for testing) */}
              <div className="flex gap-2">
                <button
                  onClick={handleSimulateHandRaise}
                  className="px-3 py-1 text-sm bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200"
                >
                  Simulate Hand Raise
                </button>
                <button
                  onClick={handleSimulateSeating}
                  className="px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
                >
                  Simulate Seating
                </button>
              </div>

              {/* Notifications */}
              <button
                onClick={toggleNotifications}
                className="relative p-2 text-gray-500 hover:text-gray-700"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                  />
                </svg>
                {notifications.length > 0 && (
                  <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-500 rounded-full">
                    {notifications.length}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('overview')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'overview'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'analytics'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Analytics
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Notifications dropdown */}
      {showNotifications && notifications.length > 0 && (
        <div className="fixed top-16 right-4 w-96 max-h-96 overflow-y-auto bg-white rounded-lg shadow-lg z-50">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Notifications</h3>
          </div>
          <div className="divide-y">
            {notifications.slice(0, 10).map((notif) => (
              <div key={notif.id} className="p-4 hover:bg-gray-50">
                <div className="flex justify-between">
                  <h4 className="font-medium text-sm">{notif.title}</h4>
                  <button
                    onClick={() => dismissNotification(notif.id)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                </div>
                <p className="text-sm text-gray-600 mt-1">{notif.message}</p>
                <span className="text-xs text-gray-400">
                  {new Date(notif.created_at).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Floor plan */}
            <div className="lg:col-span-2">
              <FloorPlan />
            </div>

            {/* Staff list */}
            <div className="lg:col-span-1">
              <StaffList />
            </div>

            {/* Event feed */}
            <div className="lg:col-span-3">
              <EventFeed />
            </div>
          </div>
        ) : (
          <Analytics />
        )}
      </main>

      {/* Error toast */}
      {connectionError && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
          {connectionError}
        </div>
      )}
    </div>
  );
}
