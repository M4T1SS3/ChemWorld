'use client'

import { useState } from 'react'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'

export default function DiscoverTab() {
  const [targetLogP, setTargetLogP] = useState(2.5)
  const [targetTPSA, setTargetTPSA] = useState(60)
  const [targetMW, setTargetMW] = useState(400)

  return (
    <div className="space-y-6 max-w-4xl">
      <Card>
        <h2 className="text-lg font-semibold mb-4">Target Properties</h2>

        <div className="space-y-6">
          {/* LogP Slider */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">LogP (Lipophilicity)</label>
              <span className="text-sm text-text-muted">{targetLogP.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="-2"
              max="6"
              step="0.1"
              value={targetLogP}
              onChange={(e) => setTargetLogP(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>-2</span>
              <span>6</span>
            </div>
          </div>

          {/* TPSA Slider */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">TPSA (Polar Surface Area)</label>
              <span className="text-sm text-text-muted">{targetTPSA.toFixed(0)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="200"
              step="5"
              value={targetTPSA}
              onChange={(e) => setTargetTPSA(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>0</span>
              <span>200</span>
            </div>
          </div>

          {/* MW Slider */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">Molecular Weight</label>
              <span className="text-sm text-text-muted">{targetMW.toFixed(0)}</span>
            </div>
            <input
              type="range"
              min="100"
              max="800"
              step="10"
              value={targetMW}
              onChange={(e) => setTargetMW(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>100</span>
              <span>800</span>
            </div>
          </div>
        </div>

        <div className="mt-6 flex gap-3">
          <Button>Start Discovery</Button>
          <Button variant="outline">Reset</Button>
        </div>
      </Card>

      {/* Results will be displayed here */}
      <div className="text-center text-text-muted py-12">
        Set target properties and click "Start Discovery" to find candidate molecules
      </div>
    </div>
  )
}
