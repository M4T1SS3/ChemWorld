#!/bin/bash

# Start ChemJEPA UI
# This script launches both frontend and backend servers

echo "🚀 Starting ChemJEPA UI..."
echo ""

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && pnpm install && cd ..
fi

# Start frontend
echo "🎨 Starting frontend on http://localhost:3000..."
cd frontend && pnpm dev &
FRONTEND_PID=$!

# Wait a moment
sleep 2

echo ""
echo "✅ ChemJEPA UI is running!"
echo "   Frontend: http://localhost:3000 (or :3001 if 3000 is busy)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for Ctrl+C
trap "kill $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 Stopped'; exit" INT
wait
