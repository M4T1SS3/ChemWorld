'use client'

import { useEffect, useRef, useState } from 'react'

interface Molecule {
  smiles: string
  embedding: number[]
  properties?: any
}

interface LatentSpaceViewerProps {
  molecules: Molecule[]
  selectedSmiles?: string
  onMoleculeClick?: (smiles: string) => void
}

export default function LatentSpaceViewer({
  molecules,
  selectedSmiles,
  onMoleculeClick
}: LatentSpaceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [projectedPoints, setProjectedPoints] = useState<Array<{ x: number; y: number; smiles: string }>>([])
  const [hoveredSmiles, setHoveredSmiles] = useState<string | null>(null)

  // Simple 2D projection using PCA-like dimensionality reduction
  useEffect(() => {
    if (molecules.length === 0) return

    // For now, use first 2 dimensions of embedding or create simple projection
    const points = molecules.map((mol, idx) => {
      // Simple projection: use first 2 dimensions or create circular layout
      const embedding = mol.embedding || []
      let x, y

      if (embedding.length >= 2) {
        // Normalize embeddings to [0, 1] range
        x = embedding[0]
        y = embedding[1]
      } else {
        // Fallback: circular layout
        const angle = (idx / molecules.length) * Math.PI * 2
        x = Math.cos(angle) * 0.5 + 0.5
        y = Math.sin(angle) * 0.5 + 0.5
      }

      return { x, y, smiles: mol.smiles }
    })

    // Normalize all points to canvas space
    if (points.length > 0) {
      const xs = points.map(p => p.x)
      const ys = points.map(p => p.y)
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs)
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys)
      const rangeX = maxX - minX || 1
      const rangeY = maxY - minY || 1

      const normalized = points.map(p => ({
        x: (p.x - minX) / rangeX,
        y: (p.y - minY) / rangeY,
        smiles: p.smiles
      }))

      setProjectedPoints(normalized)
    }
  }, [molecules])

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const padding = 40

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = '#F8FAFC'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#E2E8F0'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (width - 2 * padding)
      const y = padding + (i / 10) * (height - 2 * padding)

      // Vertical lines
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()

      // Horizontal lines
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    // Draw points
    projectedPoints.forEach((point) => {
      const x = padding + point.x * (width - 2 * padding)
      const y = padding + (1 - point.y) * (height - 2 * padding) // Flip Y axis

      const isSelected = point.smiles === selectedSmiles
      const isHovered = point.smiles === hoveredSmiles

      // Draw point
      ctx.beginPath()
      ctx.arc(x, y, isSelected ? 8 : isHovered ? 6 : 5, 0, Math.PI * 2)
      ctx.fillStyle = isSelected
        ? '#3B82F6'
        : isHovered
        ? '#60A5FA'
        : '#94A3B8'
      ctx.fill()

      // Draw border for selected
      if (isSelected) {
        ctx.strokeStyle = '#1D4ED8'
        ctx.lineWidth = 2
        ctx.stroke()
      }

      // Draw label on hover
      if (isHovered || isSelected) {
        ctx.fillStyle = '#0F172A'
        ctx.font = '12px Inter, sans-serif'
        const text = point.smiles.length > 20
          ? point.smiles.substring(0, 17) + '...'
          : point.smiles
        const textWidth = ctx.measureText(text).width

        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
        ctx.fillRect(x + 10, y - 20, textWidth + 8, 20)
        ctx.strokeStyle = '#E2E8F0'
        ctx.lineWidth = 1
        ctx.strokeRect(x + 10, y - 20, textWidth + 8, 20)

        ctx.fillStyle = '#0F172A'
        ctx.fillText(text, x + 14, y - 6)
      }
    })

    // Draw axes labels
    ctx.fillStyle = '#64748B'
    ctx.font = '14px Inter, sans-serif'
    ctx.fillText('Latent Dimension 1', width / 2 - 60, height - 10)

    ctx.save()
    ctx.translate(15, height / 2 + 60)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Latent Dimension 2', 0, 0)
    ctx.restore()

  }, [projectedPoints, selectedSmiles, hoveredSmiles])

  // Handle mouse move
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const width = canvas.width
    const height = canvas.height
    const padding = 40

    // Find nearest point
    let nearest: string | null = null
    let minDist = Infinity

    projectedPoints.forEach((point) => {
      const x = padding + point.x * (width - 2 * padding)
      const y = padding + (1 - point.y) * (height - 2 * padding)
      const dist = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2)

      if (dist < 15 && dist < minDist) {
        minDist = dist
        nearest = point.smiles
      }
    })

    setHoveredSmiles(nearest)
  }

  // Handle click
  const handleClick = () => {
    if (hoveredSmiles && onMoleculeClick) {
      onMoleculeClick(hoveredSmiles)
    }
  }

  return (
    <div className="w-full h-full bg-white rounded-xl border border-border p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text">Latent Space</h3>
        <div className="text-xs text-text-muted">
          {molecules.length} molecule{molecules.length !== 1 ? 's' : ''}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={600}
        height={400}
        className="w-full cursor-pointer"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSmiles(null)}
        onClick={handleClick}
      />

      {molecules.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-text-muted">
            <p>No molecules analyzed yet</p>
            <p className="text-xs mt-1">Analyze molecules to see them here</p>
          </div>
        </div>
      )}
    </div>
  )
}
