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
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2F3C" />
          <XAxis dataKey="name" stroke="#9CA3AF" />
          <YAxis stroke="#9CA3AF" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#151921',
              border: '1px solid #2A2F3C',
              borderRadius: '4px',
              color: '#E5E7EB',
            }}
          />
          <Bar dataKey="value" fill="#3B82F6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
