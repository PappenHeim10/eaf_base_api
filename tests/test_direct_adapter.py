import pytest
from base_api.direct_adapter import DirectMediaAdapter
from base_api.modules.errors import UnsupportedURLError
from base_api.models import MediaSource
from base_api.modules.config import DownloadConfigHLS

@pytest.fixture
def adapter():
    return DirectMediaAdapter()

def test_supports_direct_hls(adapter):
    assert adapter.supports("http://test.com/video.m3u8") is True
    assert adapter.supports("https://test.com/path/to/playlist.M3U8") is True

def test_rejects_normal_webpage(adapter):
    assert adapter.supports("http://test.com/video.html") is False
    assert adapter.supports("http://test.com/video") is False
    assert adapter.supports("ftp://test.com/video.m3u8") is False

def test_rejects_xhamster_webpage(adapter):
    assert adapter.supports("https://xhamster.com/videos/some-video-123") is False
    assert adapter.supports("https://ge.xhamster.com/moments/some-short-123") is False

@pytest.mark.asyncio
async def test_resolve_valid_hls(adapter):
    url = "https://example.com/streams/my_video_123.m3u8"
    media = await adapter.resolve(url)
    
    assert media.provider == "direct"
    assert media.original_url == url
    assert media.title == "my_video_123"
    assert len(media.sources) == 1
    
    source = media.sources[0]
    assert isinstance(source, MediaSource)
    assert source.url == url
    assert source.source_type == "HLS"

@pytest.mark.asyncio
async def test_resolve_invalid_url_raises_error(adapter):
    with pytest.raises(UnsupportedURLError):
        await adapter.resolve("https://example.com/not_an_m3u8.mp4")

@pytest.mark.asyncio
async def test_basecore_integration(adapter):
    # Verify that the resulting MediaSource can be cleanly placed into DownloadConfigHLS
    media = await adapter.resolve("https://test.com/stream.m3u8")
    
    config = DownloadConfigHLS(
        quality="1080p",
        media_source=media.sources[0],
        path="/tmp/output.mp4"
    )
    
    assert config.media_source.url == "https://test.com/stream.m3u8"
    assert config.media_source.source_type == "HLS"

def test_provider_neutrality():
    # Structural check to ensure xhamster_api is not imported in direct_adapter
    import sys
    # Import the module under test
    import base_api.direct_adapter
    
    # We can inspect the module's globals to see if xhamster_api leaked in
    module_vars = vars(base_api.direct_adapter)
    for name, obj in module_vars.items():
        # Check if the name or string representation reveals xhamster_api
        if "xhamster_api" in str(obj) or "xhamster_api" in name:
            pytest.fail(f"xhamster_api dependency found in direct_adapter: {name}")
