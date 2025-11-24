'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

interface MoleculeViewer3DThreeProps {
  smiles: string
  width?: number
  height?: number
}

// CPK atom colors (standard molecular coloring)
const ATOM_COLORS: { [key: string]: number } = {
  H: 0xFFFFFF,  // White
  C: 0x909090,  // Gray
  N: 0x3050F8,  // Blue
  O: 0xFF0D0D,  // Red
  F: 0x90E050,  // Green
  Cl: 0x1FF01F, // Green
  Br: 0xA62929, // Dark red
  I: 0x940094,  // Purple
  S: 0xFFFF30,  // Yellow
  P: 0xFF8000,  // Orange
}

const ATOM_RADII: { [key: string]: number } = {
  H: 0.25,
  C: 0.4,
  N: 0.35,
  O: 0.35,
  F: 0.3,
  Cl: 0.5,
  Br: 0.6,
  I: 0.7,
  S: 0.5,
  P: 0.5,
}

interface Atom {
  element: string
  x: number
  y: number
  z: number
}

interface Bond {
  atom1: number
  atom2: number
  order: number
}

interface MoleculeStructure {
  atoms: Atom[]
  bonds: Bond[]
}

export default function MoleculeViewer3DThree({
  smiles,
  width = 600,
  height = 400,
}: MoleculeViewer3DThreeProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const mouseRef = useRef({ x: 0, y: 0 })
  const isDraggingRef = useRef(false)

  // Fetch molecule structure from backend
  const fetchMoleculeStructure = async (smiles: string): Promise<MoleculeStructure | null> => {
    try {
      const response = await fetch('http://localhost:8001/api/molecule-structure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles }),
      })

      if (!response.ok) {
        throw new Error('Failed to fetch molecule structure')
      }

      const data = await response.json()
      return data.structure
    } catch (err) {
      console.error('Error fetching molecule structure:', err)
      // Return a simple fallback structure
      return generateFallbackStructure(smiles)
    }
  }

  // Generate a simple fallback structure for visualization
  const generateFallbackStructure = (smiles: string): MoleculeStructure => {
    // Parse SMILES to count atoms (very simplified)
    const atoms: Atom[] = []
    const bonds: Bond[] = []

    // Simple heuristic: create a linear chain of carbons
    const length = Math.min(smiles.length, 10)
    for (let i = 0; i < length; i++) {
      atoms.push({
        element: 'C',
        x: i * 1.5,
        y: Math.sin(i * 0.5) * 0.5,
        z: Math.cos(i * 0.5) * 0.5,
      })

      if (i > 0) {
        bonds.push({
          atom1: i - 1,
          atom2: i,
          order: 1,
        })
      }
    }

    return { atoms, bonds }
  }

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xffffff)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(
      45,
      width / height,
      0.1,
      1000
    )
    camera.position.set(0, 0, 15)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5)
    directionalLight1.position.set(10, 10, 10)
    scene.add(directionalLight1)

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3)
    directionalLight2.position.set(-10, -10, -10)
    scene.add(directionalLight2)

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }
    animate()

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      container.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [width, height])

  // Load and render molecule
  useEffect(() => {
    if (!smiles) return
    if (!sceneRef.current) {
      // Scene not ready yet, wait a bit
      const timer = setTimeout(() => {
        if (sceneRef.current) {
          loadMolecule()
        }
      }, 100)
      return () => clearTimeout(timer)
    }

    const loadMolecule = async () => {
      setLoading(true)
      setError(null)

      const structure = await fetchMoleculeStructure(smiles)

      if (!structure) {
        setError('Failed to load molecule')
        setLoading(false)
        return
      }

      renderMolecule(structure)
      setLoading(false)
    }

    loadMolecule()
  }, [smiles])

  const renderMolecule = (structure: MoleculeStructure) => {
    const scene = sceneRef.current
    if (!scene) return

    // Clear existing molecule
    const objectsToRemove = scene.children.filter(
      (child) => child.userData.isMolecule
    )
    objectsToRemove.forEach((obj) => {
      scene.remove(obj)
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose()
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => m.dispose())
        } else {
          obj.material.dispose()
        }
      }
    })

    // Calculate center for centering molecule
    const center = { x: 0, y: 0, z: 0 }
    structure.atoms.forEach((atom) => {
      center.x += atom.x
      center.y += atom.y
      center.z += atom.z
    })
    center.x /= structure.atoms.length
    center.y /= structure.atoms.length
    center.z /= structure.atoms.length

    // Render atoms
    structure.atoms.forEach((atom) => {
      const radius = ATOM_RADII[atom.element] || 0.4
      const color = ATOM_COLORS[atom.element] || 0x909090

      const geometry = new THREE.SphereGeometry(radius, 32, 32)
      const material = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.3,
        roughness: 0.4,
      })

      const sphere = new THREE.Mesh(geometry, material)
      sphere.position.set(
        atom.x - center.x,
        atom.y - center.y,
        atom.z - center.z
      )
      sphere.userData.isMolecule = true
      scene.add(sphere)
    })

    // Render bonds
    structure.bonds.forEach((bond) => {
      const atom1 = structure.atoms[bond.atom1]
      const atom2 = structure.atoms[bond.atom2]

      if (!atom1 || !atom2) return

      const start = new THREE.Vector3(
        atom1.x - center.x,
        atom1.y - center.y,
        atom1.z - center.z
      )
      const end = new THREE.Vector3(
        atom2.x - center.x,
        atom2.y - center.y,
        atom2.z - center.z
      )

      const direction = new THREE.Vector3().subVectors(end, start)
      const length = direction.length()
      const bondRadius = 0.1 * bond.order

      // Create cylinder for bond
      const geometry = new THREE.CylinderGeometry(
        bondRadius,
        bondRadius,
        length,
        8
      )
      const material = new THREE.MeshStandardMaterial({
        color: 0x666666,
        metalness: 0.2,
        roughness: 0.6,
      })

      const cylinder = new THREE.Mesh(geometry, material)

      // Position and orient cylinder
      const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5)
      cylinder.position.copy(midpoint)

      // Align cylinder with bond direction
      const up = new THREE.Vector3(0, 1, 0)
      const quaternion = new THREE.Quaternion().setFromUnitVectors(
        up,
        direction.normalize()
      )
      cylinder.quaternion.copy(quaternion)

      cylinder.userData.isMolecule = true
      scene.add(cylinder)
    })
  }

  // Mouse interaction
  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true
    mouseRef.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseUp = () => {
    isDraggingRef.current = false
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current || !cameraRef.current || !sceneRef.current) return

    const dx = e.clientX - mouseRef.current.x
    const dy = e.clientY - mouseRef.current.y

    const camera = cameraRef.current
    const scene = sceneRef.current

    // Rotate camera around the molecule
    const radius = camera.position.length()
    const theta = Math.atan2(camera.position.x, camera.position.z) - dx * 0.005
    const phi = Math.max(
      0.1,
      Math.min(
        Math.PI - 0.1,
        Math.acos(camera.position.y / radius) + dy * 0.005
      )
    )

    camera.position.x = radius * Math.sin(phi) * Math.sin(theta)
    camera.position.y = radius * Math.cos(phi)
    camera.position.z = radius * Math.sin(phi) * Math.cos(theta)
    camera.lookAt(0, 0, 0)

    mouseRef.current = { x: e.clientX, y: e.clientY }
  }

  const handleWheel = (e: React.WheelEvent) => {
    if (!cameraRef.current) return

    const camera = cameraRef.current
    const delta = e.deltaY * 0.01
    const newZ = Math.max(5, Math.min(50, camera.position.length() + delta))

    camera.position.multiplyScalar(newZ / camera.position.length())
  }

  if (loading) {
    return (
      <div
        className="bg-white border border-border rounded flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="text-center text-text-muted">
          <div className="animate-pulse">Loading 3D structure...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="bg-white border border-border rounded flex items-center justify-center"
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
      className="bg-white border border-border rounded relative"
      style={{ width, height, cursor: isDraggingRef.current ? 'grabbing' : 'grab' }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      <div className="absolute bottom-4 right-4 bg-white/90 border border-border rounded-lg px-3 py-2 text-xs text-text-muted">
        <div>Drag to rotate • Scroll to zoom</div>
      </div>
    </div>
  )
}
