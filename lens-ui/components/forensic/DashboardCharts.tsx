// components/forensic/DashboardCharts.tsx
"use client";
import React from "react";
import { 
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from "recharts";

interface ChartsProps {
  syntheticCount: number;
  authenticCount: number;
}

export default function DashboardCharts({ syntheticCount, authenticCount }: ChartsProps) {
  const pieData = [
    { name: "Synthetic", value: syntheticCount, color: "#EF4444" },
    { name: "Authentic", value: authenticCount, color: "#10B981" },
  ];

  const barData = [
    { name: "Mon", low: 12, medium: 8, high: 4, critical: 2 },
    { name: "Tue", low: 19, medium: 12, high: 6, critical: 1 },
    { name: "Wed", low: 15, medium: 10, high: 8, critical: 3 },
    { name: "Thu", low: 22, medium: 9, high: 5, critical: 2 },
    { name: "Fri", low: 18, medium: 14, high: 7, critical: 4 },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      
      {/* 1. Pie Chart - Detection Breakdown */}
      <div className="bg-surface border border-border p-5 rounded-xl">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          Detection Volumetrics (Authentic vs Synthetic)
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
              <Legend verticalAlign="bottom" height={36} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Stacked Bar Chart - Alerts by Confidence Band */}
      <div className="bg-surface border border-border p-5 rounded-xl">
        <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 font-mono text-gray-300">
          Alert Frequency by Confidence Band
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E1E2E" />
              <XAxis dataKey="name" stroke="#94A3B8" fontSize={11} />
              <YAxis stroke="#94A3B8" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: "#12121A", borderColor: "#1E1E2E" }} />
              <Legend />
              <Bar dataKey="low" stackId="a" fill="#10B981" name="Low" />
              <Bar dataKey="medium" stackId="a" fill="#F59E0B" name="Medium" />
              <Bar dataKey="high" stackId="a" fill="#EF4444" name="High" />
              <Bar dataKey="critical" stackId="a" fill="#991B1B" name="Critical" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
