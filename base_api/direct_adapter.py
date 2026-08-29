from urllib.parse import urlparse

from .models import Media, MediaSource
from .modules.errors import UnsupportedURLError


class DirectMediaAdapter:
    """
    Resolves direct technical media URLs (e.g. .m3u8 playlists) into Media objects.
    This adapter performs no website scraping and operates purely on the URL.
    """

    def supports(self, url: str) -> bool:
        """
        Determines if the URL is a plausible directly downloadable HLS manifest.
        This is a cheap, network-free validation.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
            
        # A simple extension check for HLS manifests
        if not parsed.path.lower().endswith('.m3u8'):
            return False
            
        return True

    async def resolve(self, url: str) -> Media:
        """
        Produces a provider-neutral Media instance containing an HLS MediaSource.
        """
        if not self.supports(url):
            raise UnsupportedURLError(f"Not a supported direct media URL: {url}")

        parsed = urlparse(url)
        
        # Fallback title strategy: extract filename from path, or use default
        filename = parsed.path.split('/')[-1]
        title = filename.rsplit('.', 1)[0] if filename else "direct_media"
        if not title:
            title = "direct_media"

        return Media(
            provider="direct",
            original_url=url,
            title=title,
            sources=[MediaSource(url=url, source_type="HLS")]
        )
