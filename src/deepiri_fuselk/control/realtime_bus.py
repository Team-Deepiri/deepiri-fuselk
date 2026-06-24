"""ZeroMQ diagnostic data bus stub."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    topic: str
    payload: np.ndarray
    timestamp: float


class RealtimeBus:
    """In-memory pub/sub bus for testing (ZMQ-compatible interface)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Frame]] = {}
        self._published: list[Frame] = []

    def publish(self, topic: str, payload: np.ndarray, timestamp: float = 0.0) -> None:
        frame = Frame(topic=topic, payload=payload, timestamp=timestamp)
        self._published.append(frame)
        for sub_topic, frames in self._subscribers.items():
            if topic.startswith(sub_topic):
                frames.append(frame)

    def subscribe(self, topic: str) -> list[Frame]:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        return self._subscribers[topic]
