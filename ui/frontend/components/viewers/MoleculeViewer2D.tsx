'use client'

import { useEffect, useRef, useState } from 'react'

interface MoleculeViewer2DProps {
  smiles: string
  width?: number
  height?: number
}

export default function MoleculeViewer2D({
  smiles,
  width = 600,
  height = 400,
}: MoleculeViewer2DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [rdkit, setRdkit] = useState<any>(null)

  // Initialize RDKit
  useEffect(() => {
    let mounted = true

    const initRDKit = async () => {
      try {
        // Dynamically import RDKit to avoid SSR issues
        const RDKitModule = await import('@rdkit/rdkit')
        // @ts-ignore - RDKit module structure
        const initRDKit = RDKitModule.initRDKitModule || RDKitModule.default

        // Configure to use WASM from public directory
        const rdkitInstance = await initRDKit({
          locateFile: (filename: string) => `/wasm/${filename}`
        })

        if (mounted) {
          setRdkit(rdkitInstance)
          setLoading(false)
        }
      } catch (err) {
        console.error('Failed to initialize RDKit:', err)
        if (mounted) {
          setError('Failed to load molecular viewer')
          setLoading(false)
        }
      }
    }

    initRDKit()

    return () => {
      mounted = false
    }
  }, [])

  // Render molecule when RDKit is ready or SMILES changes
  useEffect(() => {
    if (!rdkit || !smiles || !containerRef.current) return

    try {
      setError(null)

      // Create molecule from SMILES
      const mol = rdkit.get_mol(smiles)

      if (!mol || !mol.is_valid()) {
        setError('Invalid SMILES string')
        return
      }

      // Generate SVG
      const svg = mol.get_svg_with_highlights(
        JSON.stringify({
          width,
          height,
          bondLineWidth: 2,
          addStereoAnnotation: true,
          addAtomIndices: false,
          explicitMethyl: false,
        })
      )

      // Clear container and add SVG
      if (containerRef.current) {
        containerRef.current.innerHTML = svg
      }

      // Cleanup molecule object
      mol.delete()
    } catch (err) {
      console.error('Error rendering molecule:', err)
      setError('Failed to render molecule structure')
    }
  }, [rdkit, smiles, width, height])

  if (loading) {
    return (
      <div
        className="bg-surface border border-border rounded flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="text-center text-text-muted">
          <div className="animate-pulse">Loading 2D viewer...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="bg-surface border border-border rounded flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="text-center text-red-500">
          <p>{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="bg-white border border-border rounded flex items-center justify-center overflow-hidden"
      style={{ width, height }}
    />
  )
}
