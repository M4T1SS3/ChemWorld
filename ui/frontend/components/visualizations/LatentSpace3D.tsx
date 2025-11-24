'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

interface Molecule {
  smiles: string
  embedding: number[]
  properties?: any
}

interface LatentSpace3DProps {
  molecules: Molecule[]
  selectedSmiles?: string
  onMoleculeClick?: (smiles: string) => void
}

export default function LatentSpace3D({
  molecules,
  selectedSmiles,
  onMoleculeClick
}: LatentSpace3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const pointsRef = useRef<Map<string, THREE.Mesh>>(new Map())
  const [hoveredSmiles, setHoveredSmiles] = useState<string | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf8fafc)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    camera.position.z = 5
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4)
    directionalLight.position.set(10, 10, 10)
    scene.add(directionalLight)

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0xe2e8f0, 0xe2e8f0)
    scene.add(gridHelper)

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5)
    scene.add(axesHelper)

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate)

      // Slowly rotate camera around the scene
      camera.position.x = Math.sin(Date.now() * 0.0001) * 6
      camera.position.z = Math.cos(Date.now() * 0.0001) * 6
      camera.lookAt(0, 0, 0)

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

  // Update points when molecules change
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Clear existing points
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

    if (molecules.length === 0) return

    // Normalize embeddings to 3D space
    const embeddings3D = molecules.map((mol) => {
      const emb = mol.embedding || []
      // Use first 3 dimensions or create circular layout
      if (emb.length >= 3) {
        return {
          x: emb[0],
          y: emb[1],
          z: emb[2],
          smiles: mol.smiles
        }
      } else {
        // Circular layout for fallback
        const angle = (molecules.indexOf(mol) / molecules.length) * Math.PI * 2
        return {
          x: Math.cos(angle) * 2,
          y: Math.sin(angle) * 2,
          z: 0,
          smiles: mol.smiles
        }
      }
    })

    // Normalize to [-3, 3] range
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

    // Create spheres for each molecule
    embeddings3D.forEach(({ x, y, z, smiles }) => {
      const normalizedX = ((x - minX) / rangeX - 0.5) * 6
      const normalizedY = ((y - minY) / rangeY - 0.5) * 6
      const normalizedZ = ((z - minZ) / rangeZ - 0.5) * 6

      const isSelected = smiles === selectedSmiles
      const isHovered = smiles === hoveredSmiles

      const geometry = new THREE.SphereGeometry(
        isSelected ? 0.15 : isHovered ? 0.12 : 0.1,
        32,
        32
      )

      const material = new THREE.MeshStandardMaterial({
        color: isSelected ? 0x3b82f6 : isHovered ? 0x60a5fa : 0x94a3b8,
        emissive: isSelected ? 0x2563eb : 0x000000,
        metalness: 0.3,
        roughness: 0.4
      })

      const sphere = new THREE.Mesh(geometry, material)
      sphere.position.set(normalizedX, normalizedY, normalizedZ)
      sphere.userData = { smiles }

      scene.add(sphere)
      pointsRef.current.set(smiles, sphere)
    })

  }, [molecules, selectedSmiles, hoveredSmiles])

  // Handle mouse click
  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!cameraRef.current || !containerRef.current) return

    const container = containerRef.current
    const rect = container.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    const raycaster = new THREE.Raycaster()
    raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current)

    const meshes = Array.from(pointsRef.current.values())
    const intersects = raycaster.intersectObjects(meshes)

    if (intersects.length > 0) {
      const smiles = intersects[0].object.userData.smiles
      if (onMoleculeClick) {
        onMoleculeClick(smiles)
      }
    }
  }

  // Handle mouse move for hover
  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!cameraRef.current || !containerRef.current) return

    const container = containerRef.current
    const rect = container.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1

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
      container.style.cursor = 'default'
    }
  }

  return (
    <div className="w-full h-full bg-white rounded-xl border border-border overflow-hidden">
      <div className="p-4 border-b border-border-light flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text">3D Latent Space</h3>
        <div className="text-xs text-text-muted">
          {molecules.length} molecule{molecules.length !== 1 ? 's' : ''}
        </div>
      </div>

      <div
        ref={containerRef}
        className="w-full h-[500px] relative"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSmiles(null)}
      >
        {molecules.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-text-muted">
              <p>No molecules analyzed yet</p>
              <p className="text-xs mt-1">Analyze molecules to see them in 3D space</p>
            </div>
          </div>
        )}

        {hoveredSmiles && (
          <div className="absolute top-4 left-4 bg-white/95 border border-border rounded-lg px-3 py-2 shadow-lg">
            <div className="text-xs text-text-muted">SMILES</div>
            <div className="text-sm font-mono text-text">
              {hoveredSmiles.length > 30
                ? hoveredSmiles.substring(0, 27) + '...'
                : hoveredSmiles}
            </div>
          </div>
        )}
      </div>

      <div className="p-3 bg-surface text-xs text-text-muted border-t border-border-light">
        <div className="flex items-center justify-between">
          <span>Auto-rotating • Click to analyze • Hover for details</span>
          <span>X: Red • Y: Green • Z: Blue</span>
        </div>
      </div>
    </div>
  )
}
