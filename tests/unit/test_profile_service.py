"""Tests for profile_service.delete_profile and VectorStore.delete_collection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from core.services.profile_service import delete_profile
from rag.retriever.vector_store import VectorStore


def _chunk(source_id: str, index: int, text: str):
    return SimpleNamespace(
        id=f"{source_id}_chunk_{index}",
        source_id=source_id,
        chunk_index=index,
        section=None,
        text=text,
    )


def _count(store: VectorStore, collection_name: str) -> int:
    return len(store.get_collection(collection_name).get()["ids"])


def _db_for(profile):
    """Mock async session: profile lookup, then empty reviews, then one source row."""
    found = Mock()
    found.scalars.return_value.first.return_value = profile
    reviews = Mock()
    reviews.scalars.return_value.all.return_value = []
    sources = Mock()
    sources.scalars.return_value.all.return_value = [Mock()]
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[found, reviews, sources])
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.mark.unit
class TestDeleteProfileVectorCleanup:
    """delete_profile also removes the profile's chunks from the vector store."""

    @pytest.fixture
    def store(self, tmp_path):
        return VectorStore(persist_dir=str(tmp_path))

    @pytest.mark.asyncio
    async def test_delete_profile_removes_profile_chunks(self, store):
        profile_id = str(uuid4())
        collection = f"profile_{profile_id}"
        store.add_chunks(
            [
                (_chunk("resume_a", 0, "one"), [0.1, 0.2, 0.3]),
                (_chunk("resume_a", 1, "two"), [0.4, 0.5, 0.6]),
            ],
            collection,
        )
        assert _count(store, collection) == 2
        db = _db_for(Mock(id=profile_id))

        deleted = await delete_profile(db, profile_id, uuid4(), vector_store=store)

        assert deleted is True
        assert _count(store, collection) == 0

    @pytest.mark.asyncio
    async def test_delete_profile_leaves_other_profiles_chunks(self, store):
        profile_id = str(uuid4())
        other = f"profile_{uuid4()}"
        store.add_chunks(
            [(_chunk("resume_a", 0, "mine"), [0.1, 0.2, 0.3])], f"profile_{profile_id}"
        )
        store.add_chunks([(_chunk("resume_b", 0, "theirs"), [0.7, 0.8, 0.9])], other)

        await delete_profile(_db_for(Mock(id=profile_id)), profile_id, uuid4(), vector_store=store)

        assert _count(store, other) == 1

    @pytest.mark.asyncio
    async def test_delete_profile_not_found_makes_no_vector_call(self):
        found = Mock()
        found.scalars.return_value.first.return_value = None
        db = AsyncMock()
        db.execute = AsyncMock(return_value=found)
        vector_store = Mock()

        deleted = await delete_profile(db, uuid4(), uuid4(), vector_store=vector_store)

        assert deleted is False
        vector_store.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_profile_succeeds_when_vector_cleanup_fails(self):
        profile_id = str(uuid4())
        vector_store = Mock()
        vector_store.delete_collection.side_effect = RuntimeError("chroma down")
        db = _db_for(Mock(id=profile_id))

        deleted = await delete_profile(db, profile_id, uuid4(), vector_store=vector_store)

        assert deleted is True
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_profile_without_vector_store_skips_cleanup(self):
        profile_id = str(uuid4())

        deleted = await delete_profile(_db_for(Mock(id=profile_id)), profile_id, uuid4())

        assert deleted is True

    def test_delete_collection_missing_collection_is_a_noop(self, store):
        store.delete_collection(f"profile_{uuid4()}")
