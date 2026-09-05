import inspect
from typing import List

from base_api.models import Media
from base_api.provider import MediaProvider
from base_api.modules.errors import UnsupportedURLError, AmbiguousProviderError


class ProviderRegistry:
    """
    Registry that holds media providers and selects the appropriate one
    to resolve a given URL.
    """
    def __init__(self):
        self._providers: List[MediaProvider] = []

    def register(self, provider: MediaProvider) -> None:
        """
        Explicitly registers a new MediaProvider.
        """
        self._providers.append(provider)

    async def resolve(self, url: str) -> Media:
        """
        Finds the matching provider for a URL and resolves it into a Media object.
        Raises UnsupportedURLError if no providers match.
        Raises AmbiguousProviderError if multiple providers match.
        """
        matching_providers = [p for p in self._providers if p.supports(url)]

        if not matching_providers:
            raise UnsupportedURLError(f"No provider registered to support URL: {url}")
            
        if len(matching_providers) > 1:
            provider_names = [type(p).__name__ for p in matching_providers]
            raise AmbiguousProviderError(f"Multiple providers support URL '{url}': {', '.join(provider_names)}")
            
        provider = matching_providers[0]
        return await provider.resolve(url)

    async def close(self) -> None:
        """
        Safely shuts down all registered providers that implement an async close() method.
        Stateless providers are silently ignored.
        """
        for provider in self._providers:
            close_method = getattr(provider, "close", None)
            if close_method and inspect.iscoroutinefunction(close_method):
                await close_method()
            elif close_method and callable(close_method):
                # In case they defined a synchronous close method, though unexpected
                import asyncio
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
