// Core molecular data types
export interface MolecularProperties {
  smiles: string
  LogP?: number
  TPSA?: number
  MolWt?: number
  QED?: number
  SA?: number
  NumHDonors?: number
  NumHAcceptors?: number
  NumRotatableBonds?: number
}

export interface EnergyDecomposition {
  total: number
  binding?: number
  stability?: number
  properties?: number
  novelty?: number
}

export interface MolecularAnalysis {
  smiles: string
  properties: MolecularProperties
  energy: EnergyDecomposition
  embedding?: number[]
  uncertainty?: number
}

export interface DiscoveryCandidate {
  rank: number
  smiles: string
  score: number
  properties: MolecularProperties
  uncertainty: number
}

export interface DiscoveryConfig {
  target_logp?: number
  target_tpsa?: number
  target_mw?: number
  num_candidates?: number
  beam_size?: number
  horizon?: number
}

export interface DiscoveryTask {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress?: number
  candidates?: DiscoveryCandidate[]
  error?: string
}

export interface ComparisonData {
  molecule_a: MolecularAnalysis
  molecule_b: MolecularAnalysis
  deltas: {
    property: string
    value_a: number
    value_b: number
    delta: number
    percent_change: number
  }[]
}

export interface ModelStatus {
  encoder: 'loaded' | 'not_loaded' | 'error'
  energy: 'loaded' | 'not_loaded' | 'error'
  dynamics: 'loaded' | 'not_loaded' | 'error'
  planning: 'loaded' | 'not_loaded' | 'error'
}
