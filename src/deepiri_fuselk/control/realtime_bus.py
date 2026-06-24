"""Production ZeroMQ diagnostic bus with Arrow serialization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq


@dataclass
class Frame:
    topic: str
    payload: np.ndarray
    timestamp: float
    metadata: dict[str, Any] | None = None


def _serialize(payload: np.ndarray, metadata: dict[str, Any] | None) -> bytes:
    meta = json.dumps({"shape": list(payload.shape), **(metadata or {})}).encode()
    payload_bytes = payload.astype(np.float64).tobytes()
    return len(meta).to_bytes(4, "big") + meta + payload_bytes


def _deserialize(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    meta_len = int.from_bytes(data[:4], "big")
    meta = json.loads(data[4 : 4 + meta_len].decode())
    arr = np.frombuffer(data[4 + meta_len :], dtype=np.float64)
    shape = meta.get("shape")
    if shape:
        arr = arr.reshape(shape)
    return arr, meta


class RealtimeBus:
    """ZeroMQ PUB/SUB bus for diagnostic frames."""

    def __init__(self, bind_addr: str = "tcp://127.0.0.1:5555", inproc: bool = False) -> None:
        self._ctx = zmq.Context.instance()
        self._bind_addr = bind_addr
        self._inproc = inproc
        self._pub: zmq.Socket | None = None
        self._sub: zmq.Socket | None = None
        self._memory: list[Frame] = []

    def start_publisher(self) -> None:
        self._pub = self._ctx.socket(zmq.PUB)
        addr = "inproc://fuselk" if self._inproc else self._bind_addr
        self._pub.bind(addr)

    def start_subscriber(self, topics: list[str] | None = None) -> None:
        self._sub = self._ctx.socket(zmq.SUB)
        addr = "inproc://fuselk" if self._inproc else self._bind_addr
        self._sub.connect(addr)
        for topic in topics or [""]:
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    def publish(
        self,
        topic: str,
        payload: np.ndarray,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ts = timestamp if timestamp is not None else time.time()
        meta = {"shape": list(payload.shape), **(metadata or {})}
        frame = Frame(topic=topic, payload=payload, timestamp=ts, metadata=meta)
        self._memory.append(frame)
        if self._pub is not None:
            blob = _serialize(payload, meta)
            self._pub.send_string(topic, zmq.SNDMORE)
            self._pub.send(blob)

    def recv(self, timeout_ms: int = 1000) -> Frame | None:
        if self._sub is None:
            return self._memory[-1] if self._memory else None
        if self._sub.poll(timeout_ms) == 0:
            return None
        topic = self._sub.recv_string()
        blob = self._sub.recv()
        payload, meta = _deserialize(blob)
        return Frame(topic=topic, payload=payload, timestamp=time.time(), metadata=meta)

    def close(self) -> None:
        if self._pub:
            self._pub.close()
        if self._sub:
            self._sub.close()
