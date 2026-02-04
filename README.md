# The Reserve

Smart restaurant assistant that uses computer vision and audio to detect when customers need help and assigns the right staff member.

## What it does

- Detects hand raises and gestures using camera feeds (YOLOv8 + MediaPipe)
- Transcribes customer requests with Whisper
- Figures out who's the best staff member to handle each request
- Shows everything on a real-time dashboard

## Tech

**Backend:** Python, FastAPI, SQLite
**Frontend:** React, TypeScript, Tailwind
**ML:** YOLOv8, MediaPipe, Whisper

## Getting Started

### Docker (easiest)

```bash
docker-compose up -d
```

Dashboard: http://localhost:5173
API: http://localhost:8000/docs

### Manual

Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## How it works

1. Camera detects people and gestures at tables
2. Audio gets transcribed and classified (ordering, complaint, question, etc.)
3. System tracks table states (empty → seated → ordering → waiting → served → paying)
4. Hungarian algorithm assigns the nearest available staff
5. Dashboard shows everything in real-time via WebSocket

## Config

Create a `.env` file in backend/:

```
DEBUG=true
DATABASE_URL=sqlite+aiosqlite:///./restaurant.db
```

## Testing

The dashboard has simulation buttons to test without real cameras. You can also hit the API directly to inject events.

```bash
cd backend && pytest
```
