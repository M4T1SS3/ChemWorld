'use client'

import Card from '@/components/ui/Card'

export default function AboutTab() {
  return (
    <div className="space-y-6 max-w-4xl">
      <Card>
        <h2 className="text-xl font-semibold mb-4">ChemJEPA</h2>
        <p className="text-text-muted mb-4">
          Hierarchical Latent World Models for Molecular Discovery
        </p>

        <div className="space-y-4 text-sm">
          <div>
            <h3 className="font-medium mb-2">Novel Contributions</h3>
            <ul className="list-disc list-inside text-text-muted space-y-1">
              <li>First Latent World Model for Chemistry</li>
              <li>Hierarchical Latent Structure (z_mol → z_rxn → z_context)</li>
              <li>Learned Reaction Codebook (~1000 operators via VQ-VAE)</li>
              <li>Zero-Shot Multi-Objective Optimization</li>
              <li>Triple Uncertainty Quantification</li>
              <li>~100x Faster Planning vs SMILES MCTS</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium mb-2">Architecture</h3>
            <ul className="list-disc list-inside text-text-muted space-y-1">
              <li>E(3)-Equivariant GNN Encoder (768-dim embeddings)</li>
              <li>Decomposable Energy Model</li>
              <li>Learned Dynamics Model</li>
              <li>Novelty Detector with Triple UQ</li>
              <li>MCTS Planning Engine</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium mb-2">Model Status</h3>
            <div className="space-y-2 mt-3">
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-text-muted">Encoder</span>
                <span className="text-success font-medium">Loaded</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-text-muted">Energy Model</span>
                <span className="text-success font-medium">Loaded</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-text-muted">Dynamics Model</span>
                <span className="text-text-muted font-medium">Training</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-text-muted">Planning Engine</span>
                <span className="text-text-muted font-medium">Not Loaded</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-medium mb-2">Citation</h3>
            <div className="bg-surface p-4 rounded text-xs text-text-muted font-mono">
              @software&#123;chemjepa2025,<br />
              &nbsp;&nbsp;title=&#123;ChemJEPA: Hierarchical Latent World Models for Molecular Discovery&#125;,<br />
              &nbsp;&nbsp;year=&#123;2025&#125;,<br />
              &nbsp;&nbsp;note=&#123;First latent world model for molecular discovery with MCTS planning&#125;<br />
              &#125;
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
