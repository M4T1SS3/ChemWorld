'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { EnergyDecomposition } from '@/lib/types'

interface EnergyBarProps {
  energy: EnergyDecomposition
}

const COLORS = [
  '#2563EB', // Total - primary blue
  '#14B8A6', // Binding - teal
  '#8B5CF6', // Stability - purple
  '#10B981', // Properties - green
  '#F59E0B', // Novelty - amber
]

export default function EnergyBar({ energy }: EnergyBarProps) {
  const data = [
    { name: 'Total', value: energy.total, color: COLORS[0] },
    { name: 'Binding', value: energy.binding || 0, color: COLORS[1] },
    { name: 'Stability', value: energy.stability || 0, color: COLORS[2] },
    { name: 'Properties', value: energy.properties || 0, color: COLORS[3] },
    { name: 'Novelty', value: energy.novelty || 0, color: COLORS[4] },
  ]

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white border border-border rounded-lg shadow-lg px-3 py-2">
          <p className="text-sm font-semibold text-text mb-1">{payload[0].payload.name}</p>
          <p className="text-lg font-bold" style={{ color: payload[0].payload.color }}>
            {payload[0].value.toFixed(3)}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="w-full h-80 bg-surface rounded-xl p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
        >
          <defs>
            {data.map((entry, index) => (
              <linearGradient key={`gradient-${index}`} id={`gradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={entry.color} stopOpacity={0.9} />
                <stop offset="100%" stopColor={entry.color} stopOpacity={0.6} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#E2E8F0"
            vertical={false}
          />
          <XAxis
            dataKey="name"
            stroke="#94A3B8"
            tick={{ fill: '#475569', fontSize: 12, fontWeight: 500 }}
            axisLine={{ stroke: '#CBD5E1' }}
          />
          <YAxis
            stroke="#94A3B8"
            tick={{ fill: '#475569', fontSize: 12 }}
            axisLine={{ stroke: '#CBD5E1' }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }} />
          <Bar
            dataKey="value"
            radius={[8, 8, 0, 0]}
            animationDuration={800}
            animationBegin={0}
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={`url(#gradient-${index})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
