// API client for ChemJEPA backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'APIError'
  }
}

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new APIError(response.status, error.detail || 'Request failed')
  }

  return response.json()
}

export const api = {
  // Analyze a molecule
  analyze: async (smiles: string) => {
    return request('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({ smiles }),
    })
  },

  // Start discovery task
  discover: async (config: any) => {
    return request('/api/discover', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  // Get discovery task status
  getDiscoveryStatus: async (taskId: string) => {
    return request(`/api/discover/${taskId}`)
  },

  // Compare two molecules
  compare: async (smilesA: string, smilesB: string) => {
    return request('/api/compare', {
      method: 'POST',
      body: JSON.stringify({ smiles_a: smilesA, smiles_b: smilesB }),
    })
  },

  // Find similar molecules
  findSimilar: async (smiles: string, numResults: number = 5) => {
    return request('/api/similar', {
      method: 'POST',
      body: JSON.stringify({ smiles, num_results: numResults }),
    })
  },

  // Optimize molecule
  optimize: async (smiles: string, targetProperties: any, numCandidates: number = 10) => {
    return request('/api/optimize', {
      method: 'POST',
      body: JSON.stringify({
        smiles,
        target_properties: targetProperties,
        num_candidates: numCandidates
      }),
    })
  },

  // Get model status
  getModelStatus: async () => {
    return request('/api/models/status')
  },
}
