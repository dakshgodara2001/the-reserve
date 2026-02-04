"""Application configuration and settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Restaurant AI Assistant"
    debug: bool = True
    api_prefix: str = "/api"

    # Database
    database_url: str = "sqlite+aiosqlite:///./restaurant.db"

    # Vision settings
    yolo_model: str = "yolov8n.pt"  # nano model for speed
    detection_confidence: float = 0.5
    pose_confidence: float = 0.5

    # Audio settings
    whisper_model: str = "base"  # Options: tiny, base, small, medium, large
    sample_rate: int = 16000

    # Location simulation
    restaurant_width: float = 20.0  # meters
    restaurant_height: float = 15.0  # meters

    # Priority scoring weights
    base_wait_weight: float = 1.0
    gesture_weight: float = 30.0
    frustration_weight: float = 50.0
    verbal_request_weight: float = 40.0
    payment_ready_weight: float = 20.0

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Optional[Path] = None

    @property
    def get_data_dir(self) -> Path:
        if self.data_dir:
            return self.data_dir
        return self.base_dir / "data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
