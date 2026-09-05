import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from base_api.base import BaseCore
from base_api.modules.config import RuntimeConfig, DownloadConfigHLS
from base_api.modules.errors import MediaSourceError, UnsupportedProtocolError
from base_api.models import MediaSource

@pytest.fixture
def base_core():
    config = RuntimeConfig()
    core = BaseCore(config)
    core.threaded_download = AsyncMock(return_value=True)
    
    # Mocking cache to prevent thread lock issues or setup during tests
    core.cache = MagicMock()
    core.cache.get_segments.return_value = None
    core.get_m3u8_by_quality = AsyncMock(return_value="http://test/playlist.m3u8")
    
    return core

@pytest.mark.asyncio
async def test_base_core_download_valid_hls(base_core):
    source = MediaSource(url="http://test.com/master.m3u8", source_type="HLS")
    config = DownloadConfigHLS(quality="720p", media_source=source)
    
    result = await base_core.download(config)
    assert result is True
    
    # Assert threaded_download was called properly
    base_core.threaded_download.assert_called_once()
    kwargs = base_core.threaded_download.call_args.kwargs
    assert kwargs["pre_resolved_m3u8"] == "http://test.com/master.m3u8"
    assert kwargs["configuration"] == config

@pytest.mark.asyncio
async def test_base_core_download_missing_source(base_core):
    config = DownloadConfigHLS(quality="720p", media_source=None)
    
    with pytest.raises(MediaSourceError):
        await base_core.download(config)

@pytest.mark.asyncio
async def test_base_core_download_unsupported_protocol(base_core):
    source = MediaSource(url="http://test.com/video.mp4", source_type="HTTP")
    config = DownloadConfigHLS(quality="720p", media_source=source)
    
    with pytest.raises(UnsupportedProtocolError):
        await base_core.download(config)

@pytest.mark.asyncio
async def test_base_core_get_segments_valid_hls(base_core):
    source = MediaSource(url="http://test.com/master.m3u8", source_type="HLS")
    
    with patch("base_api.base.m3u8"): # Mock m3u8 import check
        # We need to mock get_segments internal call, or mock fetch_text
        base_core.fetch_text = AsyncMock(return_value="#EXTM3U\n#EXTINF:10,\nhttp://test.com/seg1.ts")
        
        # We just want to ensure it doesn't fail on source validation
        # It's better to just check if it gets past the validation
        try:
            await base_core.get_segments(source=source, quality="720p")
        except Exception as e:
            # We don't care about parsing errors in this test, just source validation
            assert not isinstance(e, UnsupportedProtocolError)

@pytest.mark.asyncio
async def test_base_core_get_segments_unsupported_protocol(base_core):
    source = MediaSource(url="http://test.com/master.mpd", source_type="DASH")
    
    with patch("base_api.base.m3u8"):
        with pytest.raises(UnsupportedProtocolError):
            await base_core.get_segments(source=source, quality="720p")
