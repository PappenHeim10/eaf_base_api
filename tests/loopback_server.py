"""The suite's one loopback HTTP server.

`test_hls_byterange.py` introduced a real `ThreadingHTTPServer` so that byte
ranges could be proven on the wire rather than against a mocked fetch layer.
The progressive transport needs exactly the same thing with a different
handler, so the threading, the ephemeral port and the shutdown live here once
instead of being copied: a second harness would be a second set of shutdown
bugs, and two servers that behave subtly differently would make a transport
difference look like a test difference.

Handlers stay in the test module that needs them - what is shared is how a
server is started and stopped, not what it serves.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator


class QuietHandler(BaseHTTPRequestHandler):
    """A handler that does not print a line per request into the test output.

    `protocol_version` is HTTP/1.1 on purpose: it is what real media hosts
    speak, and it is the only version where the "no Content-Length" case has to
    be expressed as chunked transfer encoding rather than as "read until the
    connection closes" - which is precisely the unknown-total case the
    progressive transport has to handle.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:  # noqa: D102 - silence stderr
        pass


@contextmanager
def serving(handler_class: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    """Run `handler_class` on an ephemeral loopback port; yield its base URL.

    Shut down deterministically: `shutdown()` stops the accept loop,
    `server_close()` releases the socket, and the thread is joined - a leaked
    server thread would keep a port and make a later test fail for no reason.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
