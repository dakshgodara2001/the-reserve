# Restaurant AI Staff Assistance System

An AI-powered system that detects customer needs via computer vision, processes audio commands, and intelligently assigns tasks to restaurant staff.

## Features

- **Computer Vision**: YOLOv8 object detection for person tracking, MediaPipe pose estimation for gesture detection (hand raises)
- **Audio Processing**: OpenAI Whisper for speech-to-text, intent classification, and menu item extraction
- **Smart Assignment**: Hungarian algorithm for optimal staff-to-task assignment
- **Real-time Dashboard**: React-based dashboard with floor plan visualization and event monitoring
- **Priority Queue**: Urgency-based task prioritization with configurable scoring

## Technology Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, WebSockets
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Zustand, Recharts
- **ML/AI**: YOLOv8 (ultralytics), MediaPipe, OpenAI Whisper, sentence-transformers
- **Database**: SQLite with async support (aiosqlite)

## Project Structure

```
restaurant-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration settings
│   │   ├── models/              # SQLAlchemy models
│   │   ├── api/routes/          # API endpoints
│   │   ├── core/                # AI engine (orchestrator, state machine, etc.)
│   │   ├── vision/              # Computer vision modules
│   │   ├── audio/               # Audio processing modules
│   │   ├── location/            # Zone management and simulation
│   │   └── services/            # Notification, analytics, sensor fusion
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── hooks/               # Custom hooks (WebSocket)
│   │   ├── stores/              # Zustand state management
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── data/
│   └── menu.json                # Sample menu for NER
├── docker-compose.yml
└── README.md
```

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and navigate to project
cd restaurant-ai

# Start all services
docker-compose up -d

# Access the dashboard at http://localhost:5173
# API docs at http://localhost:8000/docs
```

### Manual Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tables` | GET | List all tables |
| `/api/tables/{id}` | GET | Get specific table |
| `/api/tables` | POST | Create table |
| `/api/tables/{id}` | PUT | Update table |
| `/api/tables/{id}/seat` | POST | Seat customers |
| `/api/staff` | GET | List all staff |
| `/api/staff/available` | GET | Get available staff |
| `/api/staff/{id}/assign/{table_id}` | POST | Assign table to staff |
| `/api/events` | GET | Get events |
| `/api/events` | POST | Create event (testing) |
| `/api/events/{id}/resolve` | PUT | Resolve event |
| `/ws` | WebSocket | Real-time updates |

## Core Algorithms

### Priority Scoring

```python
urgency = base_wait_time_score
        + gesture_detected * 30
        + frustration_detected * 50
        + verbal_request * 40
        + payment_ready * 20
```

### Table State Machine

```
EMPTY → SEATED (person detected)
SEATED → ORDERING (menu interaction / hand raise)
ORDERING → WAITING (order placed)
WAITING → SERVED (food delivered)
SERVED → PAYING (check requested)
PAYING → EMPTY (payment complete)
```

### Staff Assignment

Uses Hungarian algorithm with cost matrix considering:
- Distance to table
- Current workload
- Skill match
- Historical response time

## Configuration

Environment variables (`.env` file):

```env
DEBUG=true
DATABASE_URL=sqlite+aiosqlite:///./restaurant.db
YOLO_MODEL=yolov8n.pt
WHISPER_MODEL=base
```

## Testing

### Backend Tests

```bash
cd backend
pytest
```

### Manual Testing

1. Start backend and frontend
2. Open dashboard at http://localhost:5173
3. Use "Simulate Hand Raise" or "Simulate Seating" buttons
4. Verify events appear in the feed
5. Check staff assignments in the staff panel

## Architecture

### Vision Pipeline

1. Video frame captured (file or webcam)
2. YOLO detects persons in frame
3. Bounding boxes passed to MediaPipe for pose estimation
4. Gestures (hand raise) detected from pose landmarks
5. Persons tracked across frames using IoU-based tracker
6. Events emitted to orchestrator

### Audio Pipeline

1. Audio captured/transcribed via Whisper
2. Intent classified (order, question, complaint, etc.)
3. Named entities extracted (menu items, quantities)
4. Request added to priority queue

### Orchestrator Flow

1. Receives events from vision/audio pipelines
2. Updates table state machines
3. Adds requests to priority queue
4. Runs assignment algorithm
5. Dispatches notifications to staff
6. Broadcasts updates via WebSocket

## Simulation Mode

Since real hardware (CCTV, BLE beacons) isn't available in prototype:

- **Video**: Supports file input or webcam
- **Audio**: Uses microphone or pre-recorded samples
- **Location**: Simulated staff movement with patrol routes
- **Events**: Manual injection via API/dashboard buttons

## Future Enhancements

- Redis/Celery for production task queue
- PostgreSQL for production database
- Real BLE/WiFi integration for staff tracking
- POS system integration
- Mobile app for staff notifications
- Voice commands via TTS earpiece integration

## License

MIT License
