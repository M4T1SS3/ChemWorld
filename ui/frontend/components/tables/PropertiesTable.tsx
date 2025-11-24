import type { MolecularProperties } from '@/lib/types'

interface PropertiesTableProps {
  properties: MolecularProperties
}

export default function PropertiesTable({ properties }: PropertiesTableProps) {
  const propertyRows = [
    { name: 'LogP', value: properties.LogP, description: 'Lipophilicity', range: { min: -2, max: 5, optimal: [0, 3] } },
    { name: 'TPSA', value: properties.TPSA, description: 'Polar Surface Area', unit: 'Ų', range: { min: 0, max: 200, optimal: [20, 130] } },
    { name: 'Mol. Weight', value: properties.MolWt, description: 'Molecular Weight', unit: 'Da', range: { min: 0, max: 1000, optimal: [150, 500] } },
    { name: 'QED', value: properties.QED, description: 'Drug-likeness', range: { min: 0, max: 1, optimal: [0.5, 1] } },
    { name: 'SA', value: properties.SA, description: 'Synthetic Accessibility', range: { min: 1, max: 10, optimal: [1, 6] } },
    { name: 'H Donors', value: properties.NumHDonors, description: 'H-Bond Donors', range: { optimal: [0, 5] } },
    { name: 'H Accept.', value: properties.NumHAcceptors, description: 'H-Bond Acceptors', range: { optimal: [0, 10] } },
    { name: 'Rotatable', value: properties.NumRotatableBonds, description: 'Rotatable Bonds', range: { optimal: [0, 10] } },
  ]

  const getValueColor = (value: number, range?: { min?: number; max?: number; optimal?: [number, number] }) => {
    if (!range?.optimal) return 'text-text'
    const [optMin, optMax] = range.optimal
    if (value >= optMin && value <= optMax) return 'text-success'
    if (range.min !== undefined && value < range.min) return 'text-error'
    if (range.max !== undefined && value > range.max) return 'text-error'
    return 'text-warning'
  }

  const getValueBg = (value: number, range?: { min?: number; max?: number; optimal?: [number, number] }) => {
    if (!range?.optimal) return 'bg-surface'
    const [optMin, optMax] = range.optimal
    if (value >= optMin && value <= optMax) return 'bg-green-50'
    if (range.min !== undefined && value < range.min) return 'bg-red-50'
    if (range.max !== undefined && value > range.max) return 'bg-red-50'
    return 'bg-amber-50'
  }

  return (
    <div className="space-y-2">
      {propertyRows.map((row) => {
        const numValue = typeof row.value === 'number' ? row.value : 0
        const displayValue = row.value !== undefined
          ? typeof row.value === 'number'
            ? row.value.toFixed(2)
            : row.value
          : 'N/A'

        return (
          <div
            key={row.name}
            className="flex items-center justify-between p-3 bg-surface hover:bg-border-light rounded-lg transition-colors group"
          >
            <div className="flex-1">
              <div className="font-medium text-sm text-text mb-0.5">{row.name}</div>
              <div className="text-xs text-text-muted">{row.description}</div>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`px-3 py-1.5 rounded-lg font-semibold text-sm ${getValueBg(numValue, row.range)} ${getValueColor(numValue, row.range)} transition-all group-hover:scale-105`}
              >
                {displayValue}
                {row.unit && <span className="text-xs ml-1 opacity-70">{row.unit}</span>}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
