'use client'

import { useEffect, useState, useRef } from 'react'
import Button from './Button'
import { SearchIcon, FlaskIcon, TargetIcon, ScaleIcon } from './Icons'

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  onAnalyze: (smiles: string) => void
  onOptimize?: () => void
  onCompare?: () => void
}

export default function CommandPalette({
  isOpen,
  onClose,
  onAnalyze,
  onOptimize,
  onCompare
}: CommandPaletteProps) {
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<'main' | 'analyze'>('main')
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  // Close on Escape
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleEscape)
      return () => window.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const exampleMolecules = [
    { name: 'Ethanol', smiles: 'CCO' },
    { name: 'Benzene', smiles: 'c1ccccc1' },
    { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  ]

  const handleSubmit = () => {
    if (input.trim()) {
      onAnalyze(input.trim())
      setInput('')
      onClose()
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 animate-fade-in"
        onClick={onClose}
      />

      {/* Command Palette */}
      <div className="fixed top-[20%] left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-4 animate-scale-in">
        <div className="bg-white rounded-2xl shadow-2xl border border-border overflow-hidden">
          {/* Search Input */}
          <div className="flex items-center gap-3 p-4 border-b border-border-light">
            <SearchIcon className="w-6 h-6 text-text-muted" />
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Enter SMILES or molecule name..."
              className="flex-1 text-lg bg-transparent border-none outline-none text-text placeholder:text-text-muted"
            />
            <kbd className="px-2 py-1 text-xs bg-surface border border-border rounded text-text-muted font-mono">
              ESC
            </kbd>
          </div>

          {/* Quick Actions */}
          {!input && (
            <div className="p-3 space-y-2">
              <div className="text-xs font-medium text-text-muted px-3 py-2">
                Quick actions
              </div>

              <button
                onClick={() => setMode('analyze')}
                className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-surface transition-colors text-left group"
              >
                <div className="w-8 h-8 rounded-lg bg-primary-50 flex items-center justify-center text-primary-600 group-hover:bg-primary-100 transition-colors">
                  <FlaskIcon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-text">Analyze molecule</div>
                  <div className="text-xs text-text-muted">Explore properties and structure</div>
                </div>
              </button>

              {onOptimize && (
                <button
                  onClick={onOptimize}
                  className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-surface transition-colors text-left group"
                >
                  <div className="w-8 h-8 rounded-lg bg-accent-teal/10 flex items-center justify-center text-accent-teal group-hover:bg-accent-teal/20 transition-colors">
                    <TargetIcon className="w-5 h-5" />
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-text">Optimize for targets</div>
                    <div className="text-xs text-text-muted">Discover new compounds</div>
                  </div>
                </button>
              )}

              {onCompare && (
                <button
                  onClick={onCompare}
                  className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-surface transition-colors text-left group"
                >
                  <div className="w-8 h-8 rounded-lg bg-accent-purple/10 flex items-center justify-center text-accent-purple group-hover:bg-accent-purple/20 transition-colors">
                    <ScaleIcon className="w-5 h-5" />
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-text">Compare molecules</div>
                    <div className="text-xs text-text-muted">Side-by-side analysis</div>
                  </div>
                </button>
              )}
            </div>
          )}

          {/* Examples */}
          {!input && (
            <div className="p-3 border-t border-border-light">
              <div className="text-xs font-medium text-text-muted px-3 py-2 mb-2">
                Quick starts
              </div>
              <div className="flex flex-wrap gap-2 px-3">
                {exampleMolecules.map((mol) => (
                  <button
                    key={mol.smiles}
                    onClick={() => {
                      setInput(mol.smiles)
                      setTimeout(() => inputRef.current?.focus(), 0)
                    }}
                    className="px-3 py-1.5 text-sm bg-primary-50 text-primary-600 rounded-lg hover:bg-primary-100 transition-colors font-medium"
                  >
                    {mol.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input preview */}
          {input && (
            <div className="p-4 border-t border-border-light bg-surface/50">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-text-muted mb-1">SMILES string</div>
                  <div className="text-sm font-mono text-text">{input}</div>
                </div>
                <Button onClick={handleSubmit} size="sm">
                  Analyze →
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Keyboard hint */}
        <div className="text-center mt-4 text-sm text-white/80 flex items-center justify-center gap-2">
          <kbd className="px-2 py-1 text-xs bg-white/10 rounded border border-white/20 font-mono">
            Cmd
          </kbd>
          <span>+</span>
          <kbd className="px-2 py-1 text-xs bg-white/10 rounded border border-white/20 font-mono">
            K
          </kbd>
          <span>to open anytime</span>
        </div>
      </div>
    </>
  )
}
