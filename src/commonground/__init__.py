"""Common Ground: a knowledge graph as a shared medium between humans and language models."""

from .consolidation import (
    DEFAULT_THRESHOLDS,
    ConsolidationPlan,
    Probe,
    Threshold,
    plan_consolidation,
)
from .graph import (
    LEVELS,
    Claim,
    CommonGround,
    Level,
    Offer,
    Participant,
    Position,
    Question,
    Source,
    Stance,
    Status,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "LEVELS",
    "Claim",
    "CommonGround",
    "ConsolidationPlan",
    "Level",
    "Offer",
    "Participant",
    "Position",
    "Probe",
    "Question",
    "Source",
    "Stance",
    "Status",
    "Threshold",
    "plan_consolidation",
]
__version__ = "0.2.0"
