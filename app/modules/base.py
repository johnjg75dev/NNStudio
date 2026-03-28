"""
app/modules/base.py
Abstract base class for every auto-discovered module type.
Concrete modules (Function, Architecture, Preset, Optimizer) all inherit from
BaseModule and implement .to_dict() for JSON serialisation to the frontend.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseModule(ABC):
    """
    Minimal contract every module must satisfy.

    Attributes
    ----------
    key         : unique snake_case identifier used in URLs and API calls
    label       : human-readable display name
    description : educational description (HTML allowed)
    category    : used to group in the UI sidebar
    """
    key:         str = ""
    label:       str = ""
    description: str = ""
    category:    str = "general"

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation sent to the frontend."""

    def __repr__(self):
        return f"<{self.__class__.__name__} key={self.key!r}>"
