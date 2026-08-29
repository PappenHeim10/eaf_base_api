from typing import Protocol, runtime_checkable
from base_api.models import Media

@runtime_checkable
class MediaProvider(Protocol):
    """
    Provider-neutral contract for media adapters.
    """
    
    def supports(self, url: str) -> bool:
        """
        Determines if the provider supports the given URL.
        Must be a cheap operation without side effects or network requests.
        """
        ...
        
    async def resolve(self, url: str) -> Media:
        """
        Resolves the given URL into a provider-neutral Media object.
        """
        ...
