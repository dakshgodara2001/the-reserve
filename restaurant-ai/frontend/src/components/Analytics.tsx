import React from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useRestaurantStore, selectTableStats, selectTotalGuests } from '../stores/restaurantStore';
import type { TableState } from '../types';

const STATE_COLORS: Record<TableState, string> = {
  empty: '#10B981',
  seated: '#3B82F6',
  ordering: '#F59E0B',
  waiting: '#8B5CF6',
  served: '#06B6D4',
  paying: '#EF4444',
};

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
}

function StatCard({ title, value, subtitle, trend, trendValue }: StatCardProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <div className="mt-2 flex items-baseline">
        <span className="text-2xl font-semibold text-gray-900">{value}</span>
        {trendValue && (
          <span
            className={`ml-2 text-sm ${
              trend === 'up'
                ? 'text-green-600'
                : trend === 'down'
                ? 'text-red-600'
                : 'text-gray-500'
            }`}
          >
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : ''} {trendValue}
          </span>
        )}
      </div>
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
    </div>
  );
}

export function Analytics() {
  const tables = useRestaurantStore((state) => state.tables);
  const staff = useRestaurantStore((state) => state.staff);
  const events = useRestaurantStore((state) => state.events);
  const tableStats = useRestaurantStore(selectTableStats);
  const totalGuests = useRestaurantStore(selectTotalGuests);

  // Calculate metrics
  const activeTables = tables.filter((t) => t.state !== 'empty').length;
  const availableStaff = staff.filter((s) => s.status === 'available').length;
  const recentEvents = events.filter((e) => {
    const eventTime = new Date(e.timestamp).getTime();
    const hourAgo = Date.now() - 3600000;
    return eventTime > hourAgo;
  });

  // Pie chart data for table states
  const tableStateData = Object.entries(tableStats)
    .filter(([_, count]) => count > 0)
    .map(([state, count]) => ({
      name: state.charAt(0).toUpperCase() + state.slice(1),
      value: count,
      color: STATE_COLORS[state as TableState],
    }));

  // Bar chart data for event types
  const eventTypeCounts: Record<string, number> = {};
  recentEvents.forEach((e) => {
    const type = e.event_type.replace(/_/g, ' ');
    eventTypeCounts[type] = (eventTypeCounts[type] || 0) + 1;
  });
  const eventBarData = Object.entries(eventTypeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name, count]) => ({ name, count }));

  // Staff workload data
  const staffWorkloadData = staff.map((s) => ({
    name: s.name.split(' ')[0],
    tables: s.assigned_tables.length,
    tasks: s.tasks_completed_today,
  }));

  // Mock hourly data (in real app, this would come from analytics service)
  const hourlyData = Array.from({ length: 12 }, (_, i) => ({
    hour: `${(i + 8).toString().padStart(2, '0')}:00`,
    guests: Math.floor(Math.random() * 30) + 10,
    events: Math.floor(Math.random() * 15) + 5,
  }));

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="Active Tables"
          value={activeTables}
          subtitle={`${tables.length} total`}
        />
        <StatCard
          title="Total Guests"
          value={totalGuests}
          trendValue="+12%"
          trend="up"
        />
        <StatCard
          title="Available Staff"
          value={availableStaff}
          subtitle={`${staff.length} total`}
        />
        <StatCard
          title="Events (1hr)"
          value={recentEvents.length}
          subtitle={`${events.filter((e) => !e.resolved).length} pending`}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Table States Pie Chart */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Table States</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={tableStateData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {tableStateData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Event Types Bar Chart */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Recent Events by Type</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={eventBarData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={100} fontSize={12} />
              <Tooltip />
              <Bar dataKey="count" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Second row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Staff Workload */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Staff Workload</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={staffWorkloadData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="tables" name="Tables" fill="#10B981" />
              <Bar dataKey="tasks" name="Tasks Today" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Trend */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Hourly Activity</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="guests"
                name="Guests"
                stroke="#3B82F6"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="events"
                name="Events"
                stroke="#F59E0B"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Overdue tables alert */}
      {tables.filter((t) => t.priority_score > 50).length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-semibold">Attention Required</h3>
          <div className="mt-2 space-y-2">
            {tables
              .filter((t) => t.priority_score > 50)
              .map((t) => (
                <div key={t.id} className="flex items-center justify-between text-sm">
                  <span>
                    Table {t.number} - {t.state}
                  </span>
                  <span className="text-red-600">Priority: {Math.round(t.priority_score)}</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
