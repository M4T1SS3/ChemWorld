'use client'

import { useEffect, useRef, useState } from 'react'

interface MoleculeViewer3DProps {
  smiles: string
  width?: number
  height?: number
}

export default function MoleculeViewer3D({
  smiles,
  width = 600,
  height = 400,
}: MoleculeViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [modules, setModules] = useState<{ $3Dmol: any; rdkit: any } | null>(null)

  // Initialize 3Dmol.js and RDKit
  useEffect(() => {
    let mounted = true

    const initModules = async () => {
      try {
        // Dynamically import to avoid SSR issues
        const [RDKitModule, mol3d] = await Promise.all([
          import('@rdkit/rdkit'),
          import('3dmol'),
        ])

        // @ts-ignore - RDKit module structure
        const initRDKit = RDKitModule.initRDKitModule || RDKitModule.default

        // Configure to use WASM from public directory
        const rdkitInstance = await initRDKit({
          locateFile: (filename: string) => `/wasm/${filename}`
        })

        if (mounted) {
          // 3Dmol exports differently - try both default and named export
          const $3Dmol = mol3d.default || mol3d
          console.log('3Dmol module:', $3Dmol)

          setModules({
            $3Dmol,
            rdkit: rdkitInstance,
          })
          setLoading(false)
        }
      } catch (err) {
        console.error('Failed to initialize 3D viewer:', err)
        if (mounted) {
          setError('Failed to load 3D molecular viewer')
          setLoading(false)
        }
      }
    }

    initModules()

    return () => {
      mounted = false
      if (viewerRef.current) {
        viewerRef.current = null
      }
    }
  }, [])

  // Render molecule in 3D
  useEffect(() => {
    if (!modules || !smiles || !containerRef.current) return

    const { $3Dmol, rdkit } = modules

    try {
      setError(null)

      // Create molecule from SMILES
      const mol = rdkit.get_mol(smiles)

      if (!mol || !mol.is_valid()) {
        setError('Invalid SMILES string')
        return
      }

      // Generate 2D coordinates (RDKit minimal doesn't have 3D embedding)
      // 3Dmol will add approximate 3D structure
      const molBlock = mol.get_molblock()

      // Clean up molecule object
      mol.delete()

      // Create or update 3Dmol viewer
      if (!viewerRef.current && containerRef.current) {
        // Clear container
        containerRef.current.innerHTML = ''

        // Create viewer
        const config = {
          backgroundColor: 'white',
        }
        viewerRef.current = $3Dmol.createViewer(containerRef.current, config)
      }

      if (viewerRef.current) {
        // Clear existing models
        viewerRef.current.clear()

        // Add molecule from molblock
        // Use 'mol' format and let 3Dmol generate approximate 3D coords
        const model = viewerRef.current.addModel(molBlock, 'mol')

        // Set style - ball and stick representation
        viewerRef.current.setStyle(
          {},
          {
            stick: {
              radius: 0.15,
              colorscheme: 'default',
            },
            sphere: {
              scale: 0.3,
              colorscheme: 'default',
            },
          }
        )

        // Add hover labels for atoms
        viewerRef.current.setHoverable(
          {},
          true,
          (atom: any) => {
            if (!atom.label) {
              viewerRef.current.addLabel(
                atom.elem,
                { position: atom, backgroundColor: 'black', fontColor: 'white' },
                { hoverable: true, hoverDuration: 0 }
              )
            }
          },
          (atom: any) => {
            if (atom.label) {
              viewerRef.current.removeLabel(atom.label)
              delete atom.label
            }
          }
        )

        // Center and zoom
        viewerRef.current.zoomTo()
        viewerRef.current.render()

        // Enable rotation
        viewerRef.current.spin(true)
      }
    } catch (err) {
      console.error('Error rendering 3D molecule:', err)
      setError('Failed to render 3D structure')
    }
  }, [modules, smiles])

  if (loading) {
    return (
      <div
        className="bg-surface border border-border rounded flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="text-center text-text-muted">
          <div className="animate-pulse">Loading 3D viewer...</div>
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
          <p className="text-xs mt-2">SMILES: {smiles}</p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="bg-white border border-border rounded"
      style={{ width, height, position: 'relative' }}
    />
  )
}
