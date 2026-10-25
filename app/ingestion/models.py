from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a human would count
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<document_id>-c<index>"
    document_id: str
    page: int
    index: int  # position within the whole document
    text: str
