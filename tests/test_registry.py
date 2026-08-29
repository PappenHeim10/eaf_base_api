import pytest
from unittest.mock import AsyncMock, MagicMock
from base_api.models import Media, MediaSource
from base_api.registry import ProviderRegistry
from base_api.modules.errors import UnsupportedURLError, AmbiguousProviderError
from base_api.provider import MediaProvider
from base_api.direct_adapter import DirectMediaAdapter

# Structurally verify that the in-tree adapter satisfies the MediaProvider protocol.
# Provider packages assert their own conformance in their own test suites; base_api
# must not depend on any of them.
def test_adapters_satisfy_protocol():
    assert isinstance(DirectMediaAdapter(), MediaProvider)

@pytest.fixture
def registry():
    return ProviderRegistry()

@pytest.mark.asyncio
async def test_registry_selects_correct_provider(registry):
    mock_media1 = Media(provider="mock1", original_url="http://test1", title="test1")
    mock_provider1 = MagicMock(spec=MediaProvider)
    mock_provider1.supports.return_value = False
    mock_provider1.resolve = AsyncMock(return_value=mock_media1)

    mock_media2 = Media(provider="mock2", original_url="http://test2", title="test2")
    mock_provider2 = MagicMock(spec=MediaProvider)
    mock_provider2.supports.side_effect = lambda url: url == "http://test2"
    mock_provider2.resolve = AsyncMock(return_value=mock_media2)

    registry.register(mock_provider1)
    registry.register(mock_provider2)

    result = await registry.resolve("http://test2")
    assert result is mock_media2
    mock_provider2.resolve.assert_called_once_with("http://test2")
    
    # Verify we didn't call resolve on the one that didn't claim the URL
    mock_provider1.resolve.assert_not_called()

@pytest.mark.asyncio
async def test_registry_raises_unsupported_url(registry):
    mock_provider = MagicMock(spec=MediaProvider)
    mock_provider.supports.return_value = False
    registry.register(mock_provider)

    with pytest.raises(UnsupportedURLError):
        await registry.resolve("http://unsupported.com")

@pytest.mark.asyncio
async def test_registry_raises_ambiguous_provider(registry):
    mock_provider1 = MagicMock(spec=MediaProvider)
    mock_provider1.supports.return_value = True
    
    mock_provider2 = MagicMock(spec=MediaProvider)
    mock_provider2.supports.return_value = True

    registry.register(mock_provider1)
    registry.register(mock_provider2)

    with pytest.raises(AmbiguousProviderError):
        await registry.resolve("http://ambiguous.com")

@pytest.mark.asyncio
async def test_registry_lifecycle_closes_stateful_providers(registry):
    # Provider with an async close
    stateful_provider = MagicMock(spec=MediaProvider)
    stateful_provider.close = AsyncMock()
    
    # Provider without close
    stateless_provider = MagicMock(spec=MediaProvider)
    del stateless_provider.close

    registry.register(stateful_provider)
    registry.register(stateless_provider)

    await registry.close()
    
    stateful_provider.close.assert_called_once()
    # Stateless provider should not have crashed the closing process

@pytest.mark.asyncio
async def test_integration_contract(registry):
    # Provide a real direct media adapter to the registry
    adapter = DirectMediaAdapter()
    registry.register(adapter)
    
    url = "https://example.com/playlist.m3u8"
    media = await registry.resolve(url)
    
    assert media.provider == "direct"
    assert len(media.sources) == 1
    
    source = media.sources[0]
    
    # Pass the source to a simulated BaseCore config to ensure boundary is respected
    from base_api.modules.config import DownloadConfigHLS
    config = DownloadConfigHLS(
        quality="1080p",
        media_source=source,
        path="/tmp/test.mp4"
    )
    
    assert config.media_source.source_type == "HLS"
    assert config.media_source.url == url
