"""Unit tests for OllamaEmbedder — mocks httpx to avoid real network calls."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


httpx = pytest.importorskip("httpx", reason="httpx not installed")

from agent_memory_manager.embedders.ollama_embedder import OllamaEmbedder


def _mock_response(status_code: int, body: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=resp
        )
    return resp


def _patch_client(response):
    """Return a context manager that patches httpx.AsyncClient.post."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=response)
    return patch("agent_memory_manager.embedders.ollama_embedder.httpx.AsyncClient",
                 return_value=mock_client)


@pytest.mark.asyncio
async def test_embed_uses_api_embed_endpoint():
    vec = [0.1, 0.2, 0.3]
    resp = _mock_response(200, {"embeddings": [vec]})
    with _patch_client(resp):
        embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=3)
        result = await embedder.embed("hello")
    assert result == vec


@pytest.mark.asyncio
async def test_embed_fallback_to_legacy_endpoint():
    vec = [0.4, 0.5, 0.6]
    resp_404 = _mock_response(404, {})
    resp_ok = _mock_response(200, {"embedding": vec})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=[resp_404, resp_ok])

    with patch("agent_memory_manager.embedders.ollama_embedder.httpx.AsyncClient",
               return_value=mock_client):
        embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=3)
        result = await embedder.embed("hello")
    assert result == vec


@pytest.mark.asyncio
async def test_dimensions_inferred_on_first_call():
    vec = [0.1] * 768
    resp = _mock_response(200, {"embeddings": [vec]})
    with _patch_client(resp):
        embedder = OllamaEmbedder(model="nomic-embed-text")
        await embedder.embed("test")
        assert embedder.dimensions == 768


@pytest.mark.asyncio
async def test_dimensions_raises_before_first_call():
    embedder = OllamaEmbedder(model="unknown-model")
    with pytest.raises(RuntimeError, match="Dimensions unknown"):
        _ = embedder.dimensions


@pytest.mark.asyncio
async def test_embed_batch_calls_embed_per_item():
    vec = [0.1, 0.2]
    resp = _mock_response(200, {"embeddings": [vec]})
    with _patch_client(resp):
        embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=2)
        results = await embedder.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert all(r == vec for r in results)


@pytest.mark.asyncio
async def test_known_model_dimensions():
    embedder = OllamaEmbedder(model="nomic-embed-text")
    assert embedder._dims == 768


@pytest.mark.asyncio
async def test_503_retries_and_eventually_raises():
    resp_503 = _mock_response(503, {})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=resp_503)

    with patch("agent_memory_manager.embedders.ollama_embedder.httpx.AsyncClient",
               return_value=mock_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):  # skip actual waits
            embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=3)
            with pytest.raises(RuntimeError, match="503"):
                await embedder.embed("test")
