'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import CommandPalette from '@/components/ui/CommandPalette'
import Button from '@/components/ui/Button'
import PropertiesTable from '@/components/tables/PropertiesTable'
import EnergyBar from '@/components/charts/EnergyBar'
import {
  BeakerIcon,
  ChartIcon,
  BoltIcon,
  LightbulbIcon,
  MoleculeIcon,
  SaveIcon,
  UploadIcon
} from '@/components/ui/Icons'

// Dynamically import viewers to avoid SSR issues
const MoleculeViewer2D = dynamic(
  () => import('@/components/viewers/MoleculeViewer2D'),
  { ssr: false }
)

const MoleculeViewer3D = dynamic(
  () => import('@/components/viewers/MoleculeViewer3DThree'),
  { ssr: false }
)

const EnergyLandscape3D = dynamic(
  () => import('@/components/visualizations/EnergyLandscape3D'),
  { ssr: false }
)

export default function Home() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [smiles, setSmiles] = useState('')
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [latentSpaceMolecules, setLatentSpaceMolecules] = useState<any[]>([])
  const [showLatentSpace, setShowLatentSpace] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [similarMolecules, setSimilarMolecules] = useState<any[]>([])
  const [loadingSimilar, setLoadingSimilar] = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [optimizationResults, setOptimizationResults] = useState<any>(null)
  const [showOptimizationMode, setShowOptimizationMode] = useState(false)

  // Cmd/Ctrl + K to open command palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setCommandPaletteOpen(true)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleAnalyze = async (smilesInput: string) => {
    setSmiles(smilesInput)
    setAnalyzing(true)
    setSidebarOpen(true)

    try {
      const response = await fetch('http://localhost:8001/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles: smilesInput }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Analysis failed')
      }

      const data = await response.json()
      setResult(data)

      // Fetch similar molecules and latent space in background
      fetchSimilarMolecules(smilesInput)
      fetchLatentSpace()
    } catch (error: any) {
      console.error('Analysis error:', error)
      alert(`Error: ${error.message}`)
    } finally {
      setAnalyzing(false)
    }
  }

  const fetchLatentSpace = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/latent-space')
      if (response.ok) {
        const data = await response.json()
        setLatentSpaceMolecules(data.molecules || [])
      }
    } catch (error) {
      console.error('Error fetching latent space:', error)
    }
  }

  const fetchSimilarMolecules = async (smilesInput: string) => {
    setLoadingSimilar(true)
    try {
      const response = await fetch('http://localhost:8001/api/similar', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles: smilesInput, num_results: 5 }),
      })

      if (response.ok) {
        const data = await response.json()
        setSimilarMolecules(data.similar_molecules || [])
      }
    } catch (error) {
      console.error('Error fetching similar molecules:', error)
    } finally {
      setLoadingSimilar(false)
    }
  }

  const handleOptimize = async () => {
    if (!smiles) {
      alert('Please analyze a molecule first')
      return
    }

    setOptimizing(true)
    setShowOptimizationMode(true)

    try {
      const response = await fetch('http://localhost:8001/api/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smiles,
          target_properties: { LogP: 2.5, TPSA: 60.0 },  // Example targets
          num_candidates: 10
        }),
      })

      if (!response.ok) {
        throw new Error('Optimization failed')
      }

      const data = await response.json()
      setOptimizationResults(data)
    } catch (error: any) {
      console.error('Optimization error:', error)
      alert(`Error: ${error.message}`)
    } finally {
      setOptimizing(false)
    }
  }

  return (
    <main className="h-screen w-screen overflow-hidden bg-surface flex flex-col">
      {/* Minimal Header */}
      <header className="flex-shrink-0 h-14 bg-white border-b border-border-light px-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BeakerIcon className="w-5 h-5 text-primary-600" />
          <h1 className="text-lg font-semibold text-text">ChemWorld</h1>
        </div>

        <div className="flex items-center gap-2">

          <button
            onClick={() => setCommandPaletteOpen(true)}
            className="px-3 py-1.5 bg-surface hover:bg-border-light rounded-lg text-sm text-text-muted transition-colors flex items-center gap-2"
          >
            <span>Search</span>
            <kbd className="px-1.5 py-0.5 bg-white border border-border rounded text-xs font-mono">
              ⌘K
            </kbd>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Central Canvas */}
        <div className="flex-1 relative flex items-center justify-center">
          {/* Empty State */}
          {!result && !analyzing && (
            <div className="text-center animate-fade-in">
              <div className="mb-6 flex justify-center">
                <BeakerIcon className="w-20 h-20 text-primary-600" />
              </div>
              <h2 className="text-2xl font-semibold text-text mb-3">
                Welcome to ChemWorld
              </h2>
              <p className="text-text-muted max-w-md mx-auto mb-6">
                Press{' '}
                <kbd className="px-2 py-1 bg-surface border border-border rounded text-xs font-mono">
                  ⌘K
                </kbd>{' '}
                to start analyzing molecules
              </p>
              <Button onClick={() => setCommandPaletteOpen(true)} size="lg">
                Open Command Palette
              </Button>
            </div>
          )}

          {/* Loading State */}
          {analyzing && (
            <div className="text-center animate-fade-in">
              <div className="w-16 h-16 mx-auto mb-4 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
              <p className="text-text-muted">Analyzing molecule...</p>
            </div>
          )}

          {/* Molecular Viewer */}
          {result && !analyzing && (
            <div className="w-full h-full flex flex-col animate-scale-in">
              {/* Viewer Container */}
              <div className="flex-1 flex items-center justify-center bg-white rounded-2xl shadow-lg border border-border p-8 overflow-hidden">
                {viewMode === '2d' ? (
                  <MoleculeViewer2D smiles={result.properties.smiles} width={900} height={700} />
                ) : (
                  <MoleculeViewer3D smiles={result.properties.smiles} width={900} height={700} />
                )}
              </div>

              {/* Floating Controls */}
              <div className="mt-4 flex justify-center gap-2">
                <div className="inline-flex gap-1 bg-white border border-border rounded-xl p-1 shadow-lg">
                  <button
                    onClick={() => setViewMode('2d')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      viewMode === '2d'
                        ? 'bg-primary-600 text-white shadow-sm'
                        : 'text-text-secondary hover:bg-surface'
                    }`}
                  >
                    2D View
                  </button>
                  <button
                    onClick={() => setViewMode('3d')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      viewMode === '3d'
                        ? 'bg-primary-600 text-white shadow-sm'
                        : 'text-text-secondary hover:bg-surface'
                    }`}
                  >
                    3D View
                  </button>
                </div>

                <button
                  onClick={() => setShowLatentSpace(!showLatentSpace)}
                  className="px-4 py-2 bg-white border border-border rounded-xl text-sm font-medium text-text-secondary hover:bg-surface transition-colors shadow-lg"
                >
                  {showLatentSpace ? 'Hide' : 'Show'} Energy Landscape
                </button>
              </div>
            </div>
          )}

          {/* Energy Landscape - Fullscreen Overlay */}
          {showLatentSpace && latentSpaceMolecules.length > 0 && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-8 animate-fade-in">
              <div className="w-full max-w-7xl h-[90vh]">
                <EnergyLandscape3D
                  molecules={latentSpaceMolecules}
                  selectedSmiles={smiles}
                  onMoleculeClick={(selectedSmiles) => {
                    handleAnalyze(selectedSmiles)
                    setShowLatentSpace(false)
                  }}
                />
              </div>
              <button
                onClick={() => setShowLatentSpace(false)}
                className="absolute top-4 right-4 w-10 h-10 bg-white/10 hover:bg-white/20 rounded-lg flex items-center justify-center text-white transition-colors"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* Right Sidebar - Analysis Panel */}
        <div
          className={`w-96 bg-white border-l border-border-light transition-all duration-300 ease-out overflow-hidden ${
            sidebarOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
          style={{ transform: sidebarOpen ? 'translateX(0)' : 'translateX(100%)' }}
        >
          {/* Sidebar Header */}
          <div className="h-14 px-4 border-b border-border-light flex items-center justify-between">
            <h3 className="font-semibold text-text">Analysis</h3>
            <button
              onClick={() => setSidebarOpen(false)}
              className="w-7 h-7 rounded-lg hover:bg-surface transition-colors flex items-center justify-center text-text-muted"
            >
              ✕
            </button>
          </div>

          {/* Sidebar Content */}
          <div className="h-[calc(100%-3.5rem)] overflow-y-auto custom-scrollbar">
            {result && (
              <div className="p-4 space-y-6">
                {/* Molecule Info */}
                <div>
                  <div className="text-xs font-medium text-text-muted mb-2">SMILES</div>
                  <div className="px-3 py-2 bg-surface rounded-lg text-sm font-mono text-text break-all">
                    {result.properties.smiles}
                  </div>
                </div>

                {/* Properties Section */}
                <div>
                  <h4 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                    <ChartIcon className="w-4 h-4 text-primary-600" />
                    Properties
                  </h4>
                  <PropertiesTable properties={result.properties} />
                </div>

                {/* Energy Section */}
                <div>
                  <h4 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                    <BoltIcon className="w-4 h-4 text-primary-600" />
                    Energy Decomposition
                  </h4>
                  <EnergyBar energy={result.energy} />
                </div>

                {/* Similar Molecules */}
                <div>
                  <h4 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                    <MoleculeIcon className="w-4 h-4 text-primary-600" />
                    Similar Molecules
                    {loadingSimilar && (
                      <span className="text-xs text-text-muted">(Loading...)</span>
                    )}
                  </h4>
                  {similarMolecules.length > 0 ? (
                    <div className="space-y-2">
                      {similarMolecules.map((mol, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleAnalyze(mol.smiles)}
                          className="w-full p-3 bg-surface rounded-lg hover:bg-border-light transition-colors text-left group"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-primary-600">
                              {(mol.similarity_score * 100).toFixed(0)}% similar
                            </span>
                            <span className="text-xs text-text-muted group-hover:text-primary-600 transition-colors">
                              Click to analyze →
                            </span>
                          </div>
                          <div className="text-xs font-mono text-text-muted truncate">
                            {mol.smiles}
                          </div>
                          <div className="flex gap-2 mt-2 text-xs text-text-muted">
                            <span>LogP: {mol.properties.LogP}</span>
                            <span>MW: {mol.properties.MolWt.toFixed(0)}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : !loadingSimilar ? (
                    <div className="text-sm text-text-muted text-center py-8 bg-surface rounded-lg">
                      {result ? 'No similar molecules found. Analyze more molecules to build the database.' : 'Analyze a molecule to find similar structures.'}
                    </div>
                  ) : (
                    <div className="text-sm text-text-muted text-center py-8 bg-surface rounded-lg">
                      Searching...
                    </div>
                  )}
                </div>

                {/* Novelty Score (Placeholder) */}
                <div>
                  <h4 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                    <LightbulbIcon className="w-4 h-4 text-primary-600" />
                    Novelty Score
                  </h4>
                  <div className="flex items-center justify-between p-3 bg-surface rounded-lg">
                    <span className="text-sm text-text-muted">Score</span>
                    <span className="text-lg font-semibold text-accent-teal">0.85</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Discovery/Optimization Modal */}
      {showOptimizationMode && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="px-6 py-4 border-b border-border-light flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-text">Molecular Optimization</h2>
                <p className="text-sm text-text-muted">Discovering optimized candidates...</p>
              </div>
              <button
                onClick={() => setShowOptimizationMode(false)}
                className="w-8 h-8 rounded-lg hover:bg-surface transition-colors flex items-center justify-center text-text-muted"
              >
                ✕
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {optimizing ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-4 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
                  <p className="text-text-muted">Running Phase 3 imagination engine...</p>
                  <p className="text-sm text-text-muted mt-2">Using counterfactual MCTS planning</p>
                </div>
              ) : optimizationResults ? (
                <div className="space-y-6">
                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-surface rounded-xl">
                      <div className="text-2xl font-bold text-primary-600">
                        {optimizationResults.candidates?.length || 0}
                      </div>
                      <div className="text-sm text-text-muted">Candidates</div>
                    </div>
                    <div className="p-4 bg-surface rounded-xl">
                      <div className="text-2xl font-bold text-accent-teal">
                        {optimizationResults.num_oracle_calls || 0}
                      </div>
                      <div className="text-sm text-text-muted">Oracle Calls</div>
                    </div>
                    <div className="p-4 bg-surface rounded-xl">
                      <div className="text-2xl font-bold text-accent-purple">
                        {optimizationResults.optimization_time?.toFixed(2) || 0}s
                      </div>
                      <div className="text-sm text-text-muted">Time</div>
                    </div>
                  </div>

                  {/* Candidates List */}
                  <div>
                    <h3 className="text-sm font-semibold text-text mb-3">Top Candidates</h3>
                    <div className="space-y-2">
                      {optimizationResults.candidates?.slice(0, 10).map((candidate: any, idx: number) => (
                        <button
                          key={idx}
                          onClick={() => {
                            handleAnalyze(candidate.smiles)
                            setShowOptimizationMode(false)
                          }}
                          className="w-full p-4 bg-surface rounded-xl hover:bg-border-light transition-colors text-left group"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-lg font-bold text-primary-600">#{candidate.rank}</span>
                              <span className="text-sm font-medium text-text">
                                Score: {candidate.score?.toFixed(3)}
                              </span>
                            </div>
                            <span className="text-xs text-text-muted group-hover:text-primary-600 transition-colors">
                              Click to analyze →
                            </span>
                          </div>
                          <div className="text-xs font-mono text-text-muted mb-2 truncate">
                            {candidate.smiles}
                          </div>
                          <div className="flex gap-3 text-xs text-text-muted">
                            <span>LogP: {candidate.properties?.LogP?.toFixed(2)}</span>
                            <span>TPSA: {candidate.properties?.TPSA?.toFixed(1)}</span>
                            <span>MW: {candidate.properties?.MolWt?.toFixed(0)}</span>
                            <span>QED: {candidate.properties?.QED?.toFixed(2)}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {/* Command Palette */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        onAnalyze={handleAnalyze}
        onOptimize={handleOptimize}
      />
    </main>
  )
}
