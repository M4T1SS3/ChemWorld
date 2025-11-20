// API client for ChemJEPA backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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

  // Get model status
  getModelStatus: async () => {
    return request('/api/models/status')
  },
}
