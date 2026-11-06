from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CollectionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    FAILED = "FAILED"


@dataclass
class CollectionResult:
    source: str
    status: CollectionStatus
    articles: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_articles(cls, source, articles):
        status = (
            CollectionStatus.SUCCESS
            if articles
            else CollectionStatus.EMPTY
        )
        return cls(source=source, status=status, articles=articles)

    @classmethod
    def failed(cls, source, error):
        return cls(
            source=source,
            status=CollectionStatus.FAILED,
            error=str(error)
        )
