"""
app/wizard_events.py
Pub/Sub event system for the wizard.
Plugins can listen to and emit events without tight coupling.
"""
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WizardEvent:
    """Represents a wizard event in the pub/sub system."""
    type: str                    # e.g., "step:changed", "validation:failed"
    step: str | None = None      # which step triggered this (e.g., "scenario", "data")
    data: Dict[str, Any] | None = None  # event payload
    timestamp: float | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().timestamp()


class EventBus:
    """
    Central event bus for wizard events.
    Plugins subscribe to events and can react to state changes.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[WizardEvent] = []

    def subscribe(self, event_type: str, callback: Callable) -> Callable:
        """
        Subscribe to an event type.
        Returns unsubscribe function for convenience.

        Usage:
            def on_step_changed(event):
                print(f"Step changed: {event.data}")

            unsub = bus.subscribe("step:changed", on_step_changed)
            unsub()  # to unsubscribe later
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)

        # Return unsubscribe function
        def unsubscribe():
            self._subscribers[event_type].remove(callback)

        return unsubscribe

    def emit(self, event: WizardEvent) -> None:
        """
        Emit an event to all subscribed listeners.
        """
        self._event_history.append(event)

        callbacks = self._subscribers.get(event.type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event handler for '{event.type}': {e}")

    def emit_simple(self, event_type: str, step: str | None = None, data: Dict[str, Any] | None = None) -> None:
        """Convenience method: create and emit an event."""
        event = WizardEvent(type=event_type, step=step, data=data)
        self.emit(event)

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_history(self) -> List[WizardEvent]:
        """Get event history (useful for debugging or replay)."""
        return self._event_history.copy()


# Global event bus instance (can be accessed by plugins)
wizard_bus = EventBus()


# Common event types (as constants for consistency)
class WizardEvents:
    """Standard event types emitted by wizard."""
    STEP_CHANGED = "step:changed"
    STEP_VALIDATED = "step:validated"
    STEP_VALIDATION_FAILED = "step:validation:failed"
    FORM_VALUE_CHANGED = "form:value:changed"
    PLUGIN_LOADED = "plugin:loaded"
    PLUGIN_ERROR = "plugin:error"
    WIZARD_COMPLETED = "wizard:completed"
    WIZARD_CANCELLED = "wizard:cancelled"
