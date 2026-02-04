import React from 'react';
import { useRestaurantStore } from '../stores/restaurantStore';
import type { Staff, StaffRole, StaffStatus } from '../types';

const ROLE_ICONS: Record<StaffRole, string> = {
  server: 'S',
  busser: 'B',
  host: 'H',
  manager: 'M',
  bartender: 'T',
};

const STATUS_COLORS: Record<StaffStatus, string> = {
  available: 'bg-green-100 text-green-800',
  busy: 'bg-yellow-100 text-yellow-800',
  on_break: 'bg-gray-100 text-gray-800',
  off_duty: 'bg-gray-200 text-gray-500',
};

const STATUS_LABELS: Record<StaffStatus, string> = {
  available: 'Available',
  busy: 'Busy',
  on_break: 'On Break',
  off_duty: 'Off Duty',
};

interface StaffCardProps {
  staff: Staff;
  isSelected: boolean;
  onClick: () => void;
}

function StaffCard({ staff, isSelected, onClick }: StaffCardProps) {
  const tables = useRestaurantStore((state) => state.tables);
  const assignedTables = tables.filter((t) => t.assigned_server_id === staff.id);

  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-all ${
        isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
      }`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Avatar */}
          <div
            className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
              staff.status === 'available' ? 'bg-green-500' : 'bg-gray-400'
            }`}
          >
            {staff.name.charAt(0)}
          </div>

          {/* Info */}
          <div>
            <h3 className="font-medium">{staff.name}</h3>
            <p className="text-sm text-gray-500 capitalize">{staff.role}</p>
          </div>
        </div>

        {/* Status badge */}
        <span className={`px-2 py-1 text-xs rounded-full ${STATUS_COLORS[staff.status]}`}>
          {STATUS_LABELS[staff.status]}
        </span>
      </div>

      {/* Details */}
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center gap-1 text-gray-600">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
          <span>
            {staff.assigned_tables.length} / {staff.max_tables} tables
          </span>
        </div>

        <div className="flex items-center gap-1 text-gray-600">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>{staff.tasks_completed_today} tasks</span>
        </div>

        <div className="flex items-center gap-1 text-gray-600">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>{Math.round(staff.avg_response_time)}s avg</span>
        </div>

        <div className="flex items-center gap-1 text-gray-600">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
            />
          </svg>
          <span className="capitalize">{staff.current_zone}</span>
        </div>
      </div>

      {/* Assigned tables */}
      {assignedTables.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {assignedTables.map((table) => (
            <span
              key={table.id}
              className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
            >
              T{table.number}
            </span>
          ))}
        </div>
      )}

      {/* Skills */}
      {staff.skills.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {staff.skills.map((skill) => (
            <span
              key={skill}
              className="px-2 py-0.5 text-xs bg-blue-50 text-blue-700 rounded"
            >
              {skill}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function StaffList() {
  const { staff, selectedStaffId, selectStaff } = useRestaurantStore();

  // Group by status
  const availableStaff = staff.filter((s) => s.status === 'available');
  const busyStaff = staff.filter((s) => s.status === 'busy');
  const otherStaff = staff.filter((s) => s.status === 'on_break' || s.status === 'off_duty');

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Staff</h2>
        <span className="text-sm text-gray-500">
          {availableStaff.length} available / {staff.length} total
        </span>
      </div>

      <div className="space-y-4">
        {/* Available */}
        {availableStaff.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-green-700 mb-2">Available</h3>
            <div className="space-y-2">
              {availableStaff.map((s) => (
                <StaffCard
                  key={s.id}
                  staff={s}
                  isSelected={selectedStaffId === s.id}
                  onClick={() => selectStaff(s.id)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Busy */}
        {busyStaff.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-yellow-700 mb-2">Busy</h3>
            <div className="space-y-2">
              {busyStaff.map((s) => (
                <StaffCard
                  key={s.id}
                  staff={s}
                  isSelected={selectedStaffId === s.id}
                  onClick={() => selectStaff(s.id)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Other */}
        {otherStaff.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Unavailable</h3>
            <div className="space-y-2">
              {otherStaff.map((s) => (
                <StaffCard
                  key={s.id}
                  staff={s}
                  isSelected={selectedStaffId === s.id}
                  onClick={() => selectStaff(s.id)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
