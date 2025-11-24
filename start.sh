#!/bin/bash
# ChemWorld Full Stack Launcher
# Starts both the FastAPI backend and Next.js frontend

set -e  # Exit on error

echo "============================================"
echo "ChemWorld - Full Stack Application"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Check if node_modules exists in frontend
if [ ! -d "ui/frontend/node_modules" ]; then
    echo -e "${RED}❌ Frontend dependencies not found!${NC}"
    echo "Please run: cd ui/frontend && pnpm install"
    exit 1
fi

# Check if models exist
if [ ! -f "checkpoints/best_jepa.pt" ]; then
    echo -e "${YELLOW}⚠️  Warning: ChemJEPA model not found at checkpoints/best_jepa.pt${NC}"
    echo "   The backend will work but predictions may be limited"
    echo ""
fi

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

# Register cleanup function for Ctrl+C
trap cleanup SIGINT SIGTERM

echo "============================================"
echo -e "${BLUE}🚀 Starting Servers...${NC}"
echo "============================================"
echo ""

# Start backend server
echo -e "${BLUE}Starting FastAPI Backend...${NC}"
source .venv/bin/activate
cd "$(dirname "$0")"
python backend/server.py > /tmp/chemworld-backend.log 2>&1 &
BACKEND_PID=$!
cd - > /dev/null

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start!${NC}"
    echo "Check logs at: /tmp/chemworld-backend.log"
    exit 1
fi

echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo ""

# Start frontend server
echo -e "${BLUE}Starting Next.js Frontend...${NC}"
cd ui/frontend
pnpm dev > /tmp/chemworld-frontend.log 2>&1 &
FRONTEND_PID=$!
cd - > /dev/null

# Wait for frontend to compile
sleep 5

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Frontend failed to start!${NC}"
    echo "Check logs at: /tmp/chemworld-frontend.log"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo ""

echo "============================================"
echo -e "${GREEN}✅ ChemWorld is now running!${NC}"
echo "============================================"
echo ""
echo "Access the application at:"
echo ""
echo -e "  ${GREEN}Frontend:${NC} http://localhost:3000 (or check logs for actual port)"
echo -e "  ${BLUE}Backend:${NC}  http://localhost:8001"
echo -e "  ${BLUE}API Docs:${NC} http://localhost:8001/docs"
echo ""
echo "Logs:"
echo -e "  ${BLUE}Backend:${NC}  /tmp/chemworld-backend.log"
echo -e "  ${BLUE}Frontend:${NC} /tmp/chemworld-frontend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo "============================================"
echo ""

# Tail both logs in the foreground
tail -f /tmp/chemworld-backend.log /tmp/chemworld-frontend.log &
TAIL_PID=$!

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID $TAIL_PID
