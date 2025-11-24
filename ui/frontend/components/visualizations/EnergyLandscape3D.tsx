'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

interface Molecule {
  smiles: string
  embedding: number[]
  properties?: any
  energy?: {
    total: number
    binding: number
    stability: number
    novelty: number
  }
}

interface OptimizationPath {
  from: string
  to: string
  steps: Array<{ embedding: number[]; energy: number }>
}

interface EnergyLandscape3DProps {
  molecules: Molecule[]
  selectedSmiles?: string
  optimizationPaths?: OptimizationPath[]
  onMoleculeClick?: (smiles: string) => void
}

export default function EnergyLandscape3D({
  molecules,
  selectedSmiles,
  optimizationPaths = [],
  onMoleculeClick
}: EnergyLandscape3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const pointsRef = useRef<Map<string, THREE.Mesh>>(new Map())
  const [hoveredSmiles, setHoveredSmiles] = useState<string | null>(null)
  const [showEnergy, setShowEnergy] = useState(true)
  const [showPaths, setShowPaths] = useState(true)
  const animationFrameRef = useRef<number | null>(null)
  const mouseRef = useRef({ x: 0, y: 0 })
  const isDraggingRef = useRef(false)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0f172a) // Dark background for energy visualization
    scene.fog = new THREE.Fog(0x0f172a, 10, 50)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100)
    camera.position.set(8, 8, 8)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambientLight)

    const directionalLight1 = new THREE.DirectionalLight(0x60a5fa, 0.6)
    directionalLight1.position.set(10, 10, 10)
    scene.add(directionalLight1)

    const directionalLight2 = new THREE.DirectionalLight(0x3b82f6, 0.4)
    directionalLight2.position.set(-10, -10, -10)
    scene.add(directionalLight2)

    // Create energy surface mesh (wireframe plane)
    const gridGeometry = new THREE.PlaneGeometry(12, 12, 20, 20)
    const gridMaterial = new THREE.MeshBasicMaterial({
      color: 0x1e40af,
      wireframe: true,
      transparent: true,
      opacity: 0.2
    })
    const gridMesh = new THREE.Mesh(gridGeometry, gridMaterial)
    gridMesh.rotation.x = -Math.PI / 2
    gridMesh.position.y = -3
    scene.add(gridMesh)

    // Axes (subtle)
    const axesHelper = new THREE.AxesHelper(6)
    axesHelper.material.transparent = true
    axesHelper.material.opacity = 0.3
    scene.add(axesHelper)

    // Animation loop
    let time = 0
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate)
      time += 0.01

      // Animate energy surface
      const positions = gridGeometry.attributes.position
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i)
        const z = positions.getZ(i)
        const wave = Math.sin(x * 0.5 + time) * Math.cos(z * 0.5 + time) * 0.3
        positions.setY(i, wave)
      }
      positions.needsUpdate = true

      renderer.render(scene, camera)
    }
    animate()

    // Handle resize
    const handleResize = () => {
      const width = container.clientWidth
      const height = container.clientHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      window.removeEventListener('resize', handleResize)
      container.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  // Energy to color mapping
  const energyToColor = (energy: number): number => {
    // Map energy to color: lower (more negative) = blue, higher = red
    const normalized = Math.max(-10, Math.min(0, energy)) / -10 // Normalize to [0, 1]

    // Gradient: Red (high energy) -> Yellow -> Green -> Blue (low energy)
    if (normalized < 0.33) {
      // Red to Yellow
      const t = normalized / 0.33
      return new THREE.Color(1, t, 0).getHex()
    } else if (normalized < 0.66) {
      // Yellow to Green
      const t = (normalized - 0.33) / 0.33
      return new THREE.Color(1 - t, 1, 0).getHex()
    } else {
      // Green to Blue
      const t = (normalized - 0.66) / 0.34
      return new THREE.Color(0, 1 - t, t).getHex()
    }
  }

  // Update points and connections
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Clear existing objects
    pointsRef.current.forEach((mesh) => {
      scene.remove(mesh)
      mesh.geometry.dispose()
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach(m => m.dispose())
      } else {
        mesh.material.dispose()
      }
    })
    pointsRef.current.clear()

    // Remove old lines
    const linesToRemove = scene.children.filter(child => child.type === 'Line')
    linesToRemove.forEach(line => {
      scene.remove(line)
      if (line instanceof THREE.Line) {
        line.geometry.dispose()
        if (Array.isArray(line.material)) {
          line.material.forEach(m => m.dispose())
        } else {
          line.material.dispose()
        }
      }
    })

    if (molecules.length === 0) return

    // Normalize embeddings
    const embeddings3D = molecules.map((mol) => {
      const emb = mol.embedding || []
      return {
        x: emb[0] || 0,
        y: emb[1] || 0,
        z: emb[2] || 0,
        smiles: mol.smiles,
        energy: mol.energy?.total || 0
      }
    })

    const xs = embeddings3D.map(p => p.x)
    const ys = embeddings3D.map(p => p.y)
    const zs = embeddings3D.map(p => p.z)

    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const minZ = Math.min(...zs)
    const maxZ = Math.max(...zs)

    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1
    const rangeZ = maxZ - minZ || 1

    // Create molecule spheres with energy coloring
    const positions = new Map<string, THREE.Vector3>()

    embeddings3D.forEach(({ x, y, z, smiles, energy }) => {
      const normalizedX = ((x - minX) / rangeX - 0.5) * 10
      const normalizedY = ((y - minY) / rangeY - 0.5) * 10
      const normalizedZ = ((z - minZ) / rangeZ - 0.5) * 10

      const position = new THREE.Vector3(normalizedX, normalizedY, normalizedZ)
      positions.set(smiles, position)

      const isSelected = smiles === selectedSmiles
      const isHovered = smiles === hoveredSmiles

      // Size based on state
      const size = isSelected ? 0.25 : isHovered ? 0.2 : 0.15

      // Geometry
      const geometry = new THREE.SphereGeometry(size, 32, 32)

      // Color based on energy
      const color = showEnergy ? energyToColor(energy) : 0x94a3b8

      const material = new THREE.MeshStandardMaterial({
        color: isSelected ? 0x60a5fa : color,
        emissive: isSelected ? 0x3b82f6 : 0x000000,
        emissiveIntensity: isSelected ? 0.5 : 0,
        metalness: 0.5,
        roughness: 0.3
      })

      const sphere = new THREE.Mesh(geometry, material)
      sphere.position.copy(position)
      sphere.userData = { smiles, energy }

      // Add glow for selected
      if (isSelected) {
        const glowGeometry = new THREE.SphereGeometry(size * 1.3, 32, 32)
        const glowMaterial = new THREE.MeshBasicMaterial({
          color: 0x60a5fa,
          transparent: true,
          opacity: 0.3
        })
        const glow = new THREE.Mesh(glowGeometry, glowMaterial)
        sphere.add(glow)
      }

      scene.add(sphere)
      pointsRef.current.set(smiles, sphere)
    })

    // Draw similarity connections
    if (showPaths && molecules.length > 1) {
      molecules.forEach((mol, i) => {
        const pos1 = positions.get(mol.smiles)
        if (!pos1) return

        // Connect to nearest neighbors
        const distances = molecules
          .filter((m, j) => j !== i)
          .map((m) => {
            const pos2 = positions.get(m.smiles)
            if (!pos2) return { smiles: m.smiles, dist: Infinity }
            return { smiles: m.smiles, dist: pos1.distanceTo(pos2) }
          })
          .sort((a, b) => a.dist - b.dist)
          .slice(0, 2) // Connect to 2 nearest neighbors

        distances.forEach(({ smiles: smiles2 }) => {
          const pos2 = positions.get(smiles2)
          if (!pos2) return

          const points = [pos1, pos2]
          const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)
          const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x475569,
            transparent: true,
            opacity: 0.2
          })
          const line = new THREE.Line(lineGeometry, lineMaterial)
          scene.add(line)
        })
      })
    }

  }, [molecules, selectedSmiles, hoveredSmiles, showEnergy, showPaths])

  // Mouse interaction
  const handleMouseDown = () => {
    isDraggingRef.current = true
  }

  const handleMouseUp = () => {
    isDraggingRef.current = false
  }

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || !cameraRef.current) return

    const container = containerRef.current
    const rect = container.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    // Rotate camera if dragging
    if (isDraggingRef.current) {
      const dx = event.clientX - mouseRef.current.x
      const dy = event.clientY - mouseRef.current.y

      const camera = cameraRef.current
      const radius = Math.sqrt(camera.position.x ** 2 + camera.position.z ** 2)
      const theta = Math.atan2(camera.position.z, camera.position.x) - dx * 0.01
      const phi = Math.max(0.1, Math.min(Math.PI - 0.1, Math.acos(camera.position.y / Math.sqrt(camera.position.x ** 2 + camera.position.y ** 2 + camera.position.z ** 2)) + dy * 0.01))

      camera.position.x = radius * Math.sin(phi) * Math.cos(theta)
      camera.position.y = radius * Math.cos(phi)
      camera.position.z = radius * Math.sin(phi) * Math.sin(theta)
      camera.lookAt(0, 0, 0)
    } else {
      // Raycasting for hover
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current)

      const meshes = Array.from(pointsRef.current.values())
      const intersects = raycaster.intersectObjects(meshes)

      if (intersects.length > 0) {
        const smiles = intersects[0].object.userData.smiles
        setHoveredSmiles(smiles)
        container.style.cursor = 'pointer'
      } else {
        setHoveredSmiles(null)
        container.style.cursor = isDraggingRef.current ? 'grabbing' : 'grab'
      }
    }

    mouseRef.current = { x: event.clientX, y: event.clientY }
  }

  const handleClick = () => {
    if (hoveredSmiles && !isDraggingRef.current && onMoleculeClick) {
      onMoleculeClick(hoveredSmiles)
    }
  }

  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">Energy Landscape</h3>
          <p className="text-xs text-slate-400">{molecules.length} molecules in latent space</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowEnergy(!showEnergy)}
            className={`px-3 py-1 rounded text-xs transition-colors ${
              showEnergy
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Energy Colors
          </button>
          <button
            onClick={() => setShowPaths(!showPaths)}
            className={`px-3 py-1 rounded text-xs transition-colors ${
              showPaths
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Connections
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="w-full h-[600px] relative"
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => {
          setHoveredSmiles(null)
          isDraggingRef.current = false
        }}
        onClick={handleClick}
      >
        {molecules.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-slate-400">
              <p className="text-lg mb-2">No molecules in latent space</p>
              <p className="text-sm">Analyze molecules to visualize the energy landscape</p>
            </div>
          </div>
        )}

        {hoveredSmiles && (
          <div className="absolute top-4 left-4 bg-slate-900/95 border border-slate-700 rounded-lg px-4 py-3 shadow-xl backdrop-blur-sm">
            <div className="text-xs text-slate-400 mb-1">SMILES</div>
            <div className="text-sm font-mono text-white mb-2">
              {hoveredSmiles.length > 40
                ? hoveredSmiles.substring(0, 37) + '...'
                : hoveredSmiles}
            </div>
            {pointsRef.current.get(hoveredSmiles)?.userData.energy !== undefined && (
              <div className="text-xs text-slate-300">
                Energy: {pointsRef.current.get(hoveredSmiles)!.userData.energy.toFixed(2)} kcal/mol
              </div>
            )}
          </div>
        )}

        <div className="absolute bottom-4 right-4 bg-slate-900/95 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300">
          <div className="font-semibold mb-1">Controls</div>
          <div>Drag to rotate • Click to analyze</div>
          {showEnergy && (
            <div className="mt-2 pt-2 border-t border-slate-700">
              <div className="font-semibold mb-1">Energy Scale</div>
              <div className="flex items-center gap-2">
                <div className="w-16 h-3 rounded" style={{ background: 'linear-gradient(to right, #ef4444, #fbbf24, #22c55e, #3b82f6)' }}></div>
                <span className="text-xs">High → Low</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
