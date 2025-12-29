"""Checkpoint history utilities for displaying conversation states.

This module provides helpers to extract human-readable summaries from
LangGraph checkpoints for the checkpoint picker UI.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CheckpointSummary:
    """Human-readable summary of a checkpoint state."""

    checkpoint_id: str
    thread_id: str
    created_at: datetime
    step: int
    human_message: str | None  # Last human message (truncated)
    ai_message: str | None  # Last AI message (truncated)
    bookmark_name: str | None  # If this checkpoint has a bookmark

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat(),
            "step": self.step,
            "human_message": self.human_message,
            "ai_message": self.ai_message,
            "bookmark_name": self.bookmark_name,
        }


def extract_last_messages(
    messages: list[dict[str, Any]], max_length: int = 100
) -> tuple[str | None, str | None]:
    """Extract last human and AI messages from a message list.

    Args:
        messages: List of message dicts with 'type' and 'content' fields.
        max_length: Maximum length for truncated message text.

    Returns:
        Tuple of (last_human_message, last_ai_message), truncated.
    """
    last_human = None
    last_ai = None

    for msg in reversed(messages):
        msg_type = msg.get("type", "")
        content = msg.get("content", "")

        # Handle content that might be a list (tool_use blocks)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        if not isinstance(content, str):
            content = str(content)

        # Truncate
        if len(content) > max_length:
            content = content[: max_length - 3] + "..."

        if msg_type == "human" and last_human is None:
            last_human = content
        elif msg_type == "ai" and last_ai is None:
            last_ai = content

        # Stop once we have both
        if last_human is not None and last_ai is not None:
            break

    return last_human, last_ai


def should_include_checkpoint(checkpoint_metadata: dict[str, Any]) -> bool:
    """Determine if a checkpoint should be shown in the picker.

    Filters out internal checkpoints (tool-only steps) to show only
    meaningful conversation states.

    Args:
        checkpoint_metadata: Checkpoint metadata from LangGraph.

    Returns:
        True if the checkpoint should be shown to users.
    """
    # Include checkpoints from human input or agent response
    source = checkpoint_metadata.get("source", "")
    # "input" = user message, "loop" = agent step, "fork" = forked state
    return source in ("input", "loop", "fork")
