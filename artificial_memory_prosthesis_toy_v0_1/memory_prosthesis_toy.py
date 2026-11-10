"""
Artificial Memory Prosthesis — toy simulation v0.1
---------------------------------------------------
A non-clinical, software-only model of an external episodic memory store
with a simulated BCI cue channel.

This does NOT decode neural signals, stimulate the brain, diagnose or treat
Alzheimer's disease, or establish that AI memory architectures transfer to
human neurobiology. The "BCI" is represented by an abstract cue vector.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional
import hashlib
import json
import math
import re


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


@dataclass
class Episode:
    episode_id: str
    timestamp: str
    person: str
    event: str
    place: str
    details: list[str]
    confidence: float = 1.0
    source: str = "caregiver-entered demonstration data"

    @property
    def text(self) -> str:
        return " ".join([self.person, self.event, self.place, *self.details])

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


class ExternalMemory:
    """Simple content-addressed episodic store with cue-based retrieval."""
    def __init__(self):
        self.records: dict[str, Episode] = {}
        self.hash_to_id: dict[str, str] = {}

    def encode(self, episode: Episode) -> str:
        digest = episode.content_hash
        if digest in self.hash_to_id:
            return self.hash_to_id[digest]
        self.records[episode.episode_id] = episode
        self.hash_to_id[digest] = episode.episode_id
        return episode.episode_id

    def retrieve(self, cue: str, top_k: int = 3,
                 available_ids: Optional[set[str]] = None) -> list[tuple[float, Episode]]:
        q = tokens(cue)
        scored = []
        for eid, ep in self.records.items():
            if available_ids is not None and eid not in available_ids:
                continue
            score = cosine(q, tokens(ep.text))
            if score > 0:
                scored.append((score, ep))
        return sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]


class SimulatedBCI:
    """
    Abstract interface: in a real research system this would be replaced by
    validated neural-signal acquisition/decoding. Here it accepts text cues.
    """
    def emit_retrieval_cue(self, cue_text: str) -> str:
        return cue_text


class MemoryProsthesisToy:
    def __init__(self):
        self.memory = ExternalMemory()
        self.bci = SimulatedBCI()
        self.available_ids: Optional[set[str]] = None

    def ingest(self, episode: Episode):
        return self.memory.encode(episode)

    def impair_encoding(self, keep_fraction: float, seed_order: list[str]):
        """Simulate missing memories by retaining only a fraction of records."""
        n = max(0, min(len(seed_order), round(len(seed_order) * keep_fraction)))
        self.available_ids = set(seed_order[:n])

    def impair_retrieval(self, cue: str, drop_terms: set[str]) -> str:
        """Simulate a degraded cue channel by removing selected cue terms."""
        return " ".join(t for t in cue.split() if t.lower() not in drop_terms)

    def assist_recall(self, cue: str, top_k: int = 3,
                      cue_drop_terms: Optional[set[str]] = None):
        neural_cue = self.bci.emit_retrieval_cue(cue)
        if cue_drop_terms:
            neural_cue = self.impair_retrieval(neural_cue, cue_drop_terms)
        matches = self.memory.retrieve(neural_cue, top_k, self.available_ids)
        return {
            "input_cue": cue,
            "simulated_bci_cue": neural_cue,
            "matches": [
                {
                    "score": round(score, 3),
                    "episode_id": ep.episode_id,
                    "timestamp": ep.timestamp,
                    "person": ep.person,
                    "event": ep.event,
                    "place": ep.place,
                    "details": ep.details,
                    "confidence": ep.confidence,
                    "source": ep.source,
                } for score, ep in matches
            ],
            "warning": (
                "Suggestions are retrieval candidates, not verified memories. "
                "The person or caregiver must confirm or correct them."
            )
        }


def demo():
    system = MemoryProsthesisToy()
    episodes = [
        Episode("E001", "2026-05-02", "Maya", "had lunch with daughter",
                "Riverside Cafe", ["daughter's name is Anika", "ordered tomato soup"]),
        Episode("E002", "2026-05-05", "Maya", "visited family doctor",
                "Green Clinic", ["appointment at 10 AM", "doctor is Dr Rao"]),
        Episode("E003", "2026-05-08", "Maya", "celebrated birthday",
                "home", ["Anika brought a yellow cake", "listened to old songs"]),
        Episode("E004", "2026-05-12", "Maya", "watered balcony plants",
                "home balcony", ["roses were flowering", "called neighbour Leena"]),
    ]
    for ep in episodes:
        system.ingest(ep)

    print("NORMAL RECALL")
    print(json.dumps(system.assist_recall("Who did I have lunch with at the cafe?"), indent=2))

    print("\\nDEGRADED CUE + EXTERNAL MEMORY SUPPORT")
    print(json.dumps(system.assist_recall(
        "Who did I have lunch with at the cafe?",
        cue_drop_terms={"lunch", "cafe"}
    ), indent=2))

    print("\\nSIMULATED ENCODING LOSS")
    system.impair_encoding(0.5, ["E001", "E002", "E003", "E004"])
    print(json.dumps(system.assist_recall("birthday cake at home"), indent=2))


if __name__ == "__main__":
    demo()
