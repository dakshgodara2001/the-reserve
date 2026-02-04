import React from 'react';
import { useRestaurantStore } from '../stores/restaurantStore';
import type { Table, Staff, TableState } from '../types';

const STATE_COLORS: Record<TableState, string> = {
  empty: '#10B981',
  seated: '#3B82F6',
  ordering: '#F59E0B',
  waiting: '#8B5CF6',
  served: '#06B6D4',
  paying: '#EF4444',
};

const STATE_LABELS: Record<TableState, string> = {
  empty: 'Empty',
  seated: 'Seated',
  ordering: 'Ordering',
  waiting: 'Waiting',
  served: 'Served',
  paying: 'Paying',
};

interface TableMarkerProps {
  table: Table;
  isSelected: boolean;
  onClick: () => void;
}

function TableMarker({ table, isSelected, onClick }: TableMarkerProps) {
  const color = STATE_COLORS[table.state];
  const isOverdue = table.priority_score > 50;

  return (
    <g
      transform={`translate(${table.x_position * 100}%, ${table.y_position * 100}%)`}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      {/* Table circle */}
      <circle
        cx="0"
        cy="0"
        r="24"
        fill={color}
        stroke={isSelected ? '#1F2937' : 'white'}
        strokeWidth={isSelected ? 3 : 2}
        className={isOverdue ? 'animate-pulse' : ''}
      />

      {/* Table number */}
      <text
        x="0"
        y="0"
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize="14"
        fontWeight="bold"
      >
        {table.number}
      </text>

      {/* Guest count badge */}
      {table.current_guests > 0 && (
        <g transform="translate(16, -16)">
          <circle cx="0" cy="0" r="10" fill="#1F2937" />
          <text
            x="0"
            y="0"
            textAnchor="middle"
            dominantBaseline="central"
            fill="white"
            fontSize="10"
            fontWeight="bold"
          >
            {table.current_guests}
          </text>
        </g>
      )}

      {/* Hand raise indicator */}
      {table.hand_raise_detected && (
        <g transform="translate(-16, -16)">
          <circle cx="0" cy="0" r="8" fill="#EF4444" className="animate-ping" />
          <text x="0" y="0" textAnchor="middle" dominantBaseline="central" fontSize="10">
            !
          </text>
        </g>
      )}
    </g>
  );
}

interface StaffMarkerProps {
  staff: Staff;
  isSelected: boolean;
  onClick: () => void;
}

function StaffMarker({ staff, isSelected, onClick }: StaffMarkerProps) {
  const statusColors: Record<string, string> = {
    available: '#10B981',
    busy: '#F59E0B',
    on_break: '#6B7280',
    off_duty: '#374151',
  };

  return (
    <g
      transform={`translate(${staff.x_position * 100}%, ${staff.y_position * 100}%)`}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      {/* Staff marker (diamond shape) */}
      <rect
        x="-12"
        y="-12"
        width="24"
        height="24"
        fill={statusColors[staff.status]}
        stroke={isSelected ? '#1F2937' : 'white'}
        strokeWidth={isSelected ? 2 : 1}
        transform="rotate(45)"
        rx="2"
      />

      {/* Initial */}
      <text
        x="0"
        y="0"
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize="10"
        fontWeight="bold"
      >
        {staff.name.charAt(0)}
      </text>
    </g>
  );
}

export function FloorPlan() {
  const { tables, staff, selectedTableId, selectedStaffId, selectTable, selectStaff } =
    useRestaurantStore();

  const zones = [
    { id: 'entrance', name: 'Entrance', x: 0, y: 0, width: 15, height: 20, color: '#E5E7EB' },
    { id: 'main', name: 'Main Dining', x: 15, y: 0, width: 55, height: 70, color: '#FEF3C7' },
    { id: 'bar', name: 'Bar', x: 0, y: 70, width: 50, height: 30, color: '#DBEAFE' },
    { id: 'private', name: 'Private', x: 70, y: 0, width: 30, height: 50, color: '#FCE7F3' },
    { id: 'kitchen', name: 'Kitchen', x: 70, y: 50, width: 30, height: 50, color: '#FEE2E2' },
  ];

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Floor Plan</h2>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mb-4 text-sm">
        {Object.entries(STATE_LABELS).map(([state, label]) => (
          <div key={state} className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: STATE_COLORS[state as TableState] }}
            />
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Floor plan SVG */}
      <div className="relative w-full aspect-[4/3] border border-gray-200 rounded-lg overflow-hidden">
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="xMidYMid meet"
          className="w-full h-full"
        >
          {/* Zones */}
          {zones.map((zone) => (
            <g key={zone.id}>
              <rect
                x={zone.x}
                y={zone.y}
                width={zone.width}
                height={zone.height}
                fill={zone.color}
                stroke="#D1D5DB"
                strokeWidth="0.5"
              />
              <text
                x={zone.x + zone.width / 2}
                y={zone.y + 5}
                textAnchor="middle"
                fontSize="3"
                fill="#6B7280"
              >
                {zone.name}
              </text>
            </g>
          ))}

          {/* Tables */}
          {tables.map((table) => (
            <g
              key={table.id}
              transform={`translate(${table.x_position * 100}, ${table.y_position * 100})`}
              onClick={() => selectTable(table.id)}
              className="cursor-pointer"
            >
              <circle
                r="4"
                fill={STATE_COLORS[table.state]}
                stroke={selectedTableId === table.id ? '#1F2937' : 'white'}
                strokeWidth={selectedTableId === table.id ? 0.5 : 0.3}
                className={table.priority_score > 50 ? 'animate-pulse' : ''}
              />
              <text
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize="2.5"
                fontWeight="bold"
              >
                {table.number}
              </text>
              {table.current_guests > 0 && (
                <g transform="translate(3, -3)">
                  <circle r="1.5" fill="#1F2937" />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="white"
                    fontSize="1.5"
                  >
                    {table.current_guests}
                  </text>
                </g>
              )}
            </g>
          ))}

          {/* Staff */}
          {staff.map((s) => (
            <g
              key={s.id}
              transform={`translate(${s.x_position * 100}, ${s.y_position * 100})`}
              onClick={() => selectStaff(s.id)}
              className="cursor-pointer"
            >
              <rect
                x="-2"
                y="-2"
                width="4"
                height="4"
                fill={s.status === 'available' ? '#10B981' : '#F59E0B'}
                stroke={selectedStaffId === s.id ? '#1F2937' : 'white'}
                strokeWidth={selectedStaffId === s.id ? 0.4 : 0.2}
                transform="rotate(45)"
              />
              <text
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize="2"
                fontWeight="bold"
              >
                {s.name.charAt(0)}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Selected info */}
      {selectedTableId && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          {(() => {
            const table = tables.find((t) => t.id === selectedTableId);
            if (!table) return null;
            return (
              <div>
                <h3 className="font-semibold">Table {table.number}</h3>
                <div className="text-sm text-gray-600 mt-1">
                  <p>State: {STATE_LABELS[table.state]}</p>
                  <p>Guests: {table.current_guests} / {table.capacity}</p>
                  <p>Zone: {table.zone}</p>
                  {table.assigned_server_id && (
                    <p>
                      Server:{' '}
                      {staff.find((s) => s.id === table.assigned_server_id)?.name || 'Unknown'}
                    </p>
                  )}
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
