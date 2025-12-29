"""Bookmark storage for conversation checkpoints.

This module provides a BookmarkManager that stores named references to LangGraph checkpoints.
Bookmarks allow users to save conversation states and resume from them later.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg


@dataclass
class Bookmark:
    """A saved reference to a conversation checkpoint."""

    id: str
    thread_id: str
    checkpoint_id: str
    name: str | None
    description: str | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }


class BookmarkManager:
    """Manage conversation bookmarks in PostgreSQL.

    Uses psycopg3 for async PostgreSQL operations. Bookmarks are stored
    in a dedicated table and reference LangGraph checkpoints.

    Usage:
        manager = BookmarkManager(postgres_uri)
        await manager.initialize()  # Creates table if needed

        # Save a bookmark
        bookmark = await manager.save(thread_id, checkpoint_id, name="my-save")

        # List bookmarks
        bookmarks = await manager.list(thread_id)

        # Resume from bookmark
        bookmark = await manager.get("my-save")  # by name
        bookmark = await manager.get("abc-123")  # or by ID
    """

    TABLE_NAME = "conversation_bookmarks"

    def __init__(self, postgres_uri: str):
        """Initialize with PostgreSQL connection string.

        Args:
            postgres_uri: PostgreSQL connection URI (same as used for LangGraph).
        """
        self.postgres_uri = postgres_uri

    async def initialize(self) -> None:
        """Create the bookmarks table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
            id VARCHAR(36) PRIMARY KEY,
            thread_id VARCHAR(255) NOT NULL,
            checkpoint_id VARCHAR(255) NOT NULL,
            name VARCHAR(255),
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT unique_bookmark_name UNIQUE (name)
        );
        CREATE INDEX IF NOT EXISTS idx_bookmarks_thread_id 
            ON {self.TABLE_NAME} (thread_id);
        CREATE INDEX IF NOT EXISTS idx_bookmarks_name 
            ON {self.TABLE_NAME} (name) WHERE name IS NOT NULL;
        """
        async with await psycopg.AsyncConnection.connect(self.postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(create_table_sql)
            await conn.commit()

    async def save(
        self,
        thread_id: str,
        checkpoint_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Bookmark:
        """Save a bookmark to a checkpoint.

        Args:
            thread_id: The LangGraph thread ID.
            checkpoint_id: The checkpoint ID to bookmark.
            name: Optional human-readable name (must be unique if provided).
            description: Optional description of what's saved.

        Returns:
            The created Bookmark object.

        Raises:
            ValueError: If a bookmark with the given name already exists.
        """
        bookmark_id = str(uuid4())
        created_at = datetime.now(timezone.utc)

        insert_sql = f"""
        INSERT INTO {self.TABLE_NAME} (id, thread_id, checkpoint_id, name, description, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        try:
            async with await psycopg.AsyncConnection.connect(
                self.postgres_uri
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        insert_sql,
                        (
                            bookmark_id,
                            thread_id,
                            checkpoint_id,
                            name,
                            description,
                            created_at,
                        ),
                    )
                await conn.commit()
        except psycopg.errors.UniqueViolation as e:
            raise ValueError(f"Bookmark with name '{name}' already exists") from e

        return Bookmark(
            id=bookmark_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            name=name,
            description=description,
            created_at=created_at,
        )

    async def get(self, identifier: str) -> Bookmark | None:
        """Get a bookmark by name or ID.

        Args:
            identifier: Either the bookmark name or ID.

        Returns:
            The Bookmark if found, None otherwise.
        """
        # Try to find by name first, then by ID
        select_sql = f"""
        SELECT id, thread_id, checkpoint_id, name, description, created_at
        FROM {self.TABLE_NAME}
        WHERE name = %s OR id = %s
        LIMIT 1
        """

        async with await psycopg.AsyncConnection.connect(self.postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(select_sql, (identifier, identifier))
                row = await cur.fetchone()

        if row is None:
            return None

        return Bookmark(
            id=row[0],
            thread_id=row[1],
            checkpoint_id=row[2],
            name=row[3],
            description=row[4],
            created_at=row[5],
        )

    async def list(self, thread_id: str | None = None) -> list[Bookmark]:
        """List bookmarks, optionally filtered by thread.

        Args:
            thread_id: Optional thread ID to filter by.

        Returns:
            List of Bookmark objects, most recent first.
        """
        if thread_id:
            select_sql = f"""
            SELECT id, thread_id, checkpoint_id, name, description, created_at
            FROM {self.TABLE_NAME}
            WHERE thread_id = %s
            ORDER BY created_at DESC
            """
            params = (thread_id,)
        else:
            select_sql = f"""
            SELECT id, thread_id, checkpoint_id, name, description, created_at
            FROM {self.TABLE_NAME}
            ORDER BY created_at DESC
            """
            params = ()

        bookmarks = []
        async with await psycopg.AsyncConnection.connect(self.postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(select_sql, params)
                rows = await cur.fetchall()

        for row in rows:
            bookmarks.append(
                Bookmark(
                    id=row[0],
                    thread_id=row[1],
                    checkpoint_id=row[2],
                    name=row[3],
                    description=row[4],
                    created_at=row[5],
                )
            )

        return bookmarks

    async def delete(self, identifier: str) -> bool:
        """Delete a bookmark by name or ID.

        Args:
            identifier: Either the bookmark name or ID.

        Returns:
            True if a bookmark was deleted, False if not found.
        """
        delete_sql = f"""
        DELETE FROM {self.TABLE_NAME}
        WHERE name = %s OR id = %s
        """

        async with await psycopg.AsyncConnection.connect(self.postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(delete_sql, (identifier, identifier))
                deleted = cur.rowcount > 0
            await conn.commit()

        return deleted
