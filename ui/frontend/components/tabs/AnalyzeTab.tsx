'use client'

import { useState } from 'react'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import PropertiesTable from '@/components/tables/PropertiesTable'
import EnergyBar from '@/components/charts/EnergyBar'

export default function AnalyzeTab() {
  const [smiles, setSmiles] = useState('')
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleAnalyze = async () => {
    if (!smiles) return

    setAnalyzing(true)
    // TODO: Call API
    setTimeout(() => {
      // Mock data for now
      setResult({
        properties: {
          smiles,
          LogP: 2.5,
          TPSA: 60,
          MolWt: 180,
          QED: 0.72,
          SA: 2.1,
          NumHDonors: 2,
          NumHAcceptors: 3,
          NumRotatableBonds: 1,
        },
        energy: {
          total: -5.2,
          binding: -2.1,
          stability: -1.5,
          properties: -1.2,
          novelty: -0.4,
        },
      })
      setAnalyzing(false)
    }, 1000)
  }

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <Card>
        <h2 className="text-lg font-semibold mb-4">Molecule Input</h2>

        <div>
          <label className="block text-sm font-medium mb-2">
            SMILES String
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
              placeholder="e.g., CCO (ethanol)"
              className="flex-1 px-3 py-2 border border-border rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <Button onClick={handleAnalyze} disabled={!smiles || analyzing}>
              {analyzing ? 'Analyzing...' : 'Analyze'}
            </Button>
          </div>

          <p className="text-xs text-text-muted mt-2">
            Enter a SMILES string to analyze molecular properties and energy decomposition
          </p>
        </div>
      </Card>

      {result && (
        <>
          {/* Molecular Viewer */}
          <Card>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Structure</h2>

              {/* 2D/3D Toggle */}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={viewMode === '2d' ? 'primary' : 'outline'}
                  onClick={() => setViewMode('2d')}
                >
                  2D
                </Button>
                <Button
                  size="sm"
                  variant={viewMode === '3d' ? 'primary' : 'outline'}
                  onClick={() => setViewMode('3d')}
                >
                  3D
                </Button>
              </div>
            </div>

            {/* Placeholder for molecular viewer */}
            <div className="bg-white border border-border rounded h-96 flex items-center justify-center">
              <div className="text-center text-text-muted">
                <p className="text-lg mb-2">{viewMode.toUpperCase()} Molecular Viewer</p>
                <p className="text-sm">Viewer will be integrated here</p>
                <p className="text-xs mt-2">SMILES: {result.properties.smiles}</p>
              </div>
            </div>
          </Card>

          {/* Properties Table */}
          <Card>
            <h2 className="text-lg font-semibold mb-4">Molecular Properties</h2>
            <PropertiesTable properties={result.properties} />
          </Card>

          {/* Energy Decomposition */}
          <Card>
            <h2 className="text-lg font-semibold mb-4">Energy Decomposition</h2>
            <EnergyBar energy={result.energy} />
          </Card>
        </>
      )}

      {!result && (
        <div className="text-center text-text-muted py-12">
          Enter a SMILES string above to analyze a molecule
        </div>
      )}
    </div>
  )
}
