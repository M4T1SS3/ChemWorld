'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { EnergyDecomposition } from '@/lib/types'

interface EnergyBarProps {
  energy: EnergyDecomposition
}

export default function EnergyBar({ energy }: EnergyBarProps) {
  const data = [
    { name: 'Total', value: energy.total },
    { name: 'Binding', value: energy.binding || 0 },
    { name: 'Stability', value: energy.stability || 0 },
    { name: 'Properties', value: energy.properties || 0 },
    { name: 'Novelty', value: energy.novelty || 0 },
  ]

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis dataKey="name" stroke="#6B7280" />
          <YAxis stroke="#6B7280" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#F8F9FA',
              border: '1px solid #E5E7EB',
              borderRadius: '4px',
            }}
          />
          <Bar dataKey="value" fill="#2563EB" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
