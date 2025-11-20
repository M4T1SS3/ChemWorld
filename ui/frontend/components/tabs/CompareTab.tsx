'use client'

import { useState } from 'react'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'

export default function CompareTab() {
  const [smilesA, setSmilesA] = useState('')
  const [smilesB, setSmilesB] = useState('')

  return (
    <div className="space-y-6 max-w-6xl">
      <Card>
        <h2 className="text-lg font-semibold mb-4">Compare Molecules</h2>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Molecule A (SMILES)
            </label>
            <input
              type="text"
              value={smilesA}
              onChange={(e) => setSmilesA(e.target.value)}
              placeholder="e.g., CCO"
              className="w-full px-3 py-2 border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Molecule B (SMILES)
            </label>
            <input
              type="text"
              value={smilesB}
              onChange={(e) => setSmilesB(e.target.value)}
              placeholder="e.g., CCCO"
              className="w-full px-3 py-2 border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        </div>

        <Button className="mt-4">Compare</Button>
      </Card>

      {/* Comparison results will be displayed here */}
      <div className="text-center text-text-muted py-12">
        Enter two SMILES strings to compare molecules
      </div>
    </div>
  )
}
