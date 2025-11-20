# ChemJEPA UI - Clean Scientific Interface

A minimal, scientific web interface for ChemJEPA molecular discovery system.

## Design Principles

- **Clean & Minimal** - White backgrounds, generous whitespace
- **Scientific** - Data-focused, no marketing fluff
- **Familiar** - Looks like research tools (AlphaFold, PubChem)
- **Fast** - Instant interactions, responsive

## Tech Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts (simple bar/scatter)
- **3D Viewer**: Three.js + @react-three/fiber
- **2D Viewer**: RDKit.js (WASM)
- **HTTP**: SWR (caching + revalidation)

### Backend
- **Framework**: FastAPI
- **Models**: ChemJEPA (from ../chemjepa/)
- **Async**: Background tasks
- **CORS**: Enabled for localhost:3000

## Quick Start

### 1. Setup Frontend

```bash
cd frontend

# Install dependencies (using pnpm)
pnpm install

# Or use npm
npm install

# Run development server
pnpm dev
```

Frontend runs on `http://localhost:3000`

### 2. Setup Backend

```bash
cd ../backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Backend API runs on `http://localhost:8000`

### 3. Open Browser

Navigate to `http://localhost:3000`

##Project Structure

```
chemjepa-ui/
├── frontend/                    # Next.js app
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Main tabbed interface
│   │   └── globals.css         # Tailwind + custom styles
│   │
│   ├── components/
│   │   ├── tabs/               # Tab components
│   │   │   ├── AnalyzeTab.tsx
│   │   │   ├── DiscoverTab.tsx
│   │   │   ├── CompareTab.tsx
│   │   │   └── AboutTab.tsx
│   │   │
│   │   ├── molecular/          # Molecular viewers
│   │   │   ├── MolecularViewer.tsx
│   │   │   ├── Viewer2D.tsx
│   │   │   └── Viewer3D.tsx
│   │   │
│   │   ├── inputs/             # Form inputs
│   │   │   ├── SMILESInput.tsx
│   │   │   └── PropertySliders.tsx
│   │   │
│   │   ├── tables/             # Data tables
│   │   │   ├── PropertiesTable.tsx
│   │   │   ├── ResultsTable.tsx
│   │   │   └── ComparisonTable.tsx
│   │   │
│   │   ├── charts/             # Charts
│   │   │   ├── EnergyBar.tsx
│   │   │   └── PropertyScatter.tsx
│   │   │
│   │   └── ui/                 # Base components
│   │       ├── Button.tsx
│   │       ├── Card.tsx
│   │       ├── Tabs.tsx
│   │       └── Table.tsx
│   │
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   ├── types.ts            # TypeScript types
│   │   └── utils.ts            # Utilities
│   │
│   ├── hooks/
│   │   ├── useAnalyze.ts
│   │   └── useDiscover.ts
│   │
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.js
│
├── backend/                     # FastAPI app
│   ├── main.py                 # FastAPI entry
│   ├── api/
│   │   ├── analyze.py          # Molecule analysis
│   │   └── discover.py         # MCTS discovery
│   │
│   ├── services/
│   │   ├── molecular.py
│   │   └── planning.py
│   │
│   ├── requirements.txt
│   └── config.py
│
└── README.md (this file)
```

## Color Palette

```css
/* Scientific Minimal */
Background:  #FFFFFF (pure white)
Surface:     #F8F9FA (light gray cards)
Border:      #E5E7EB (subtle borders)
Text:        #1F2937 (dark gray)
Text-Muted:  #6B7280 (lighter gray)

/* Accents (minimal use) */
Primary:     #2563EB (blue)
Success:     #059669 (green)
Warning:     #D97706 (amber)
Error:       #DC2626 (red)
```

## UI Pages

### 1. Analyze Tab

**Clean layout:**
- SMILES input with validation
- 2D/3D molecular viewer (toggle button)
- Properties table (simple data table)
- Energy decomposition (bar chart)
- Export buttons (CSV, PNG)

**No:**
- Fancy animations
- Pie charts
- Complex visualizations

### 2. Discover Tab

**Focused on results:**
- Target property sliders
- MCTS configuration inputs
- Results table (rank, score, properties)
- Export functionality

**No:**
- MCTS tree visualization (too complex)
- Just show final results table

### 3. Compare Tab

**Side-by-side comparison:**
- Select 2 molecules (A vs B)
- Show structures side-by-side
- Table with property differences
- Export comparison

### 4. About Tab

**System information:**
- Model status (loaded/not loaded)
- Brief description
- Citation info
- Dataset stats

## API Endpoints

### Analyze
```
POST /api/analyze
Body: { smiles: string }
Returns: { properties, energy, embedding }
```

### Discover
```
POST /api/discover
Body: { target_logp, target_tpsa, target_mw, config }
Returns: { task_id }

GET /api/discover/{task_id}
Returns: { status, progress, candidates }
```

### Compare
```
POST /api/compare
Body: { smiles_a, smiles_b }
Returns: { comparison_data }
```

### Models
```
GET /api/models/status
Returns: { encoder: "loaded", energy: "loaded", planning: "loaded" }
```

## Development Checklist

### Phase 1: Foundation (4 hours)
- [ ] Setup Next.js with TypeScript
- [ ] Setup Tailwind CSS
- [ ] Create FastAPI backend
- [ ] Basic /api/analyze endpoint
- [ ] Test API connection

### Phase 2: Analyze Tab (8 hours)
- [ ] SMILES input component
- [ ] 2D viewer (RDKit.js)
- [ ] 3D viewer (Three.js)
- [ ] Toggle between 2D/3D
- [ ] Properties table
- [ ] Energy bar chart
- [ ] Export functionality

### Phase 3: Discover Tab (8 hours)
- [ ] Property sliders
- [ ] MCTS config inputs
- [ ] /api/discover endpoint
- [ ] Status polling
- [ ] Results table
- [ ] Export results

### Phase 4: Polish (4 hours)
- [ ] Compare tab
- [ ] About tab
- [ ] Loading states
- [ ] Error handling
- [ ] Responsive design

**Total:** ~24 hours for complete UI

## Deployment

### Docker Compose (Recommended)

```bash
# Create docker-compose.yml
docker-compose up
```

**Services:**
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

### Manual

**Frontend (Vercel):**
```bash
cd frontend
vercel deploy
```

**Backend (Railway/Fly.io):**
```bash
cd backend
railway up
# or
fly deploy
```

## Next Steps

1. **Install dependencies** and run dev servers
2. **Implement Analyze tab** first (most used)
3. **Add Discover tab** second
4. **Polish** with Compare/About tabs
5. **Deploy** when ready

## Notes

- Keep it simple and clean
- Focus on functionality over fancy UI
- Make it fast and responsive
- Easy to export data (CSV, PNG)
- Clear, readable typography
- Generous whitespace

---

**Timeline:** 2-3 weeks for production-ready UI
**Complexity:** Medium (React + FastAPI)
**Result:** Clean, scientific interface suitable for Stanford presentations
