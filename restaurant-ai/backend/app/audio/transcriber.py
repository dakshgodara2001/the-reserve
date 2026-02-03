"""Audio transcription using OpenAI Whisper."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import numpy as np


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    confidence: float
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "segment_count": len(self.segments),
            "metadata": self.metadata,
        }


class AudioTranscriber:
    """
    Audio transcription using OpenAI Whisper.

    Supports:
    - Audio file transcription
    - Real-time audio stream transcription
    - Multiple model sizes for speed/accuracy tradeoff
    """

    SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: Optional[str] = "en",
    ):
        """
        Initialize transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            language: Language code or None for auto-detection
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._initialized = False

    def initialize(self):
        """Load Whisper model."""
        if self._initialized:
            return

        try:
            import whisper

            print(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            self._initialized = True
            print("Whisper model loaded successfully")
        except ImportError:
            print("Warning: openai-whisper not installed, using mock transcriber")
            self._initialized = True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self._initialized = True

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for Whisper

        Returns:
            TranscriptionResult
        """
        if not self._initialized:
            self.initialize()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {audio_path.suffix}")

        if self.model is None:
            return self._mock_transcribe(str(audio_path))

        # Transcribe
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            **kwargs,
        )

        # Calculate confidence from segment probabilities
        segments = result.get("segments", [])
        if segments:
            avg_prob = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            confidence = 1.0 - avg_prob
        else:
            confidence = 0.8

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", self.language or "en"),
            confidence=confidence,
            duration_seconds=segments[-1]["end"] if segments else 0.0,
            segments=[
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for s in segments
            ],
        )

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio data directly.

        Args:
            audio_data: Audio as numpy array (float32, -1 to 1)
            sample_rate: Sample rate of audio
            **kwargs: Additional arguments for Whisper

        Returns:
            TranscriptionResult
        """
        if not self._initialized:
            self.initialize()

        if self.model is None:
            return self._mock_transcribe("audio_data")

        # Ensure correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize if needed
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            try:
                import librosa

                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000
                )
            except ImportError:
                print("Warning: librosa not available for resampling")

        # Transcribe
        result = self.model.transcribe(
            audio_data,
            language=self.language,
            **kwargs,
        )

        segments = result.get("segments", [])
        if segments:
            avg_prob = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            confidence = 1.0 - avg_prob
        else:
            confidence = 0.8

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", self.language or "en"),
            confidence=confidence,
            duration_seconds=len(audio_data) / 16000,
            segments=[
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for s in segments
            ],
        )

    async def transcribe_stream(
        self,
        audio_stream,
        chunk_duration: float = 5.0,
        overlap: float = 0.5,
    ):
        """
        Transcribe audio stream in real-time.

        Args:
            audio_stream: Async generator yielding audio chunks
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks for continuity

        Yields:
            TranscriptionResult for each chunk
        """
        import asyncio

        if not self._initialized:
            self.initialize()

        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(chunk_duration * 16000)
        overlap_samples = int(overlap * 16000)

        async for chunk in audio_stream:
            # Add to buffer
            buffer = np.concatenate([buffer, chunk])

            # Process when we have enough samples
            while len(buffer) >= chunk_samples:
                # Extract chunk
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples - overlap_samples :]

                # Transcribe
                result = self.transcribe_audio(audio_chunk)

                if result.text.strip():
                    yield result

                await asyncio.sleep(0)

        # Process remaining buffer
        if len(buffer) > 16000:  # At least 1 second
            result = self.transcribe_audio(buffer)
            if result.text.strip():
                yield result

    def _mock_transcribe(self, source: str) -> TranscriptionResult:
        """Generate mock transcription for testing."""
        import random

        mock_phrases = [
            "Excuse me, can I get some water please?",
            "We're ready to order.",
            "Could I see the dessert menu?",
            "Can we get the check please?",
            "This steak is excellent, thank you.",
            "Excuse me, we've been waiting for a while.",
            "Could I get a refill on this coffee?",
            "What do you recommend for appetizers?",
        ]

        text = random.choice(mock_phrases)

        return TranscriptionResult(
            text=text,
            language="en",
            confidence=random.uniform(0.7, 0.95),
            duration_seconds=random.uniform(2.0, 5.0),
            metadata={"mock": True, "source": source},
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "zh",
            "ja",
            "ko",
            "ru",
            "ar",
            "hi",
        ]
