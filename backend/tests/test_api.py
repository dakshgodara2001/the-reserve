"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_root_endpoint():
    """Test root endpoint returns app info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["status"] == "running"


@pytest.mark.anyio
async def test_health_check():
    """Test health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_get_tables():
    """Test getting all tables."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/tables")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_get_staff():
    """Test getting all staff."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/staff")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_get_events():
    """Test getting events."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/events")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_create_event():
    """Test creating an event."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/events",
            json={
                "event_type": "hand_raise",
                "priority": "high",
                "table_id": 1,
                "description": "Test hand raise",
                "source": "test",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["event_type"] == "hand_raise"
    assert data["table_id"] == 1


@pytest.mark.anyio
async def test_get_event_stats():
    """Test getting event statistics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/events/stats")

    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "by_type" in data


@pytest.mark.anyio
async def test_invalid_table_state():
    """Test updating table with invalid state."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            "/api/tables/1",
            json={"state": "invalid_state"},
        )

    # Should return 400 for invalid state
    assert response.status_code == 400


@pytest.mark.anyio
async def test_get_available_staff():
    """Test getting available staff."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/staff/available")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # All returned staff should be available
    for staff in data:
        assert staff["status"] == "available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
