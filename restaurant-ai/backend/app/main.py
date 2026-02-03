"""FastAPI application entry point for Restaurant AI Assistant."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import init_db
from .api.routes import tables, staff, events, websocket
from .services.notification import notification_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print(f"Starting {settings.app_name}...")

    # Initialize database
    await init_db()
    print("Database initialized")

    # Initialize sample data if database is empty
    await initialize_sample_data()

    # Start background services
    notification_task = asyncio.create_task(notification_service.start())

    yield

    # Shutdown
    print("Shutting down...")
    notification_task.cancel()
    try:
        await notification_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.app_name,
    description="AI-powered restaurant staff assistance system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tables.router, prefix=settings.api_prefix)
app.include_router(staff.router, prefix=settings.api_prefix)
app.include_router(events.router, prefix=settings.api_prefix)
app.include_router(websocket.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


async def initialize_sample_data():
    """Initialize sample tables and staff if database is empty."""
    from sqlalchemy import select
    from .models import Table, Staff, StaffRole, async_session

    async with async_session() as db:
        # Check if tables exist
        result = await db.execute(select(Table).limit(1))
        if result.scalar_one_or_none() is None:
            # Create sample tables
            sample_tables = [
                Table(number=1, capacity=2, x_position=0.1, y_position=0.2, zone="window"),
                Table(number=2, capacity=4, x_position=0.3, y_position=0.2, zone="window"),
                Table(number=3, capacity=4, x_position=0.5, y_position=0.2, zone="main"),
                Table(number=4, capacity=6, x_position=0.7, y_position=0.2, zone="main"),
                Table(number=5, capacity=2, x_position=0.1, y_position=0.5, zone="main"),
                Table(number=6, capacity=4, x_position=0.3, y_position=0.5, zone="main"),
                Table(number=7, capacity=4, x_position=0.5, y_position=0.5, zone="main"),
                Table(number=8, capacity=8, x_position=0.7, y_position=0.5, zone="private"),
                Table(number=9, capacity=2, x_position=0.1, y_position=0.8, zone="bar"),
                Table(number=10, capacity=2, x_position=0.3, y_position=0.8, zone="bar"),
            ]

            for table in sample_tables:
                db.add(table)

            print("Created 10 sample tables")

        # Check if staff exist
        result = await db.execute(select(Staff).limit(1))
        if result.scalar_one_or_none() is None:
            # Create sample staff
            sample_staff = [
                Staff(
                    name="Alice Johnson",
                    role=StaffRole.SERVER,
                    skills=["fine_dining", "wine"],
                    max_tables=4,
                    x_position=0.2,
                    y_position=0.3,
                ),
                Staff(
                    name="Bob Smith",
                    role=StaffRole.SERVER,
                    skills=["speed", "large_parties"],
                    max_tables=5,
                    x_position=0.4,
                    y_position=0.4,
                ),
                Staff(
                    name="Carol Davis",
                    role=StaffRole.SERVER,
                    skills=["cocktails", "desserts"],
                    max_tables=4,
                    x_position=0.6,
                    y_position=0.3,
                ),
                Staff(
                    name="David Wilson",
                    role=StaffRole.BUSSER,
                    skills=["quick_turnaround"],
                    max_tables=8,
                    x_position=0.8,
                    y_position=0.5,
                ),
                Staff(
                    name="Emma Brown",
                    role=StaffRole.HOST,
                    skills=["reservations", "vip"],
                    max_tables=0,
                    x_position=0.05,
                    y_position=0.1,
                ),
                Staff(
                    name="Frank Miller",
                    role=StaffRole.BARTENDER,
                    skills=["cocktails", "speed"],
                    max_tables=0,
                    x_position=0.2,
                    y_position=0.85,
                    current_zone="bar",
                ),
            ]

            for staff_member in sample_staff:
                db.add(staff_member)

            print("Created 6 sample staff members")

        await db.commit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
