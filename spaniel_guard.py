"""Local, fail-closed inspection before a text retrieval tool returns content.

Standard library only. Run from a SPANIEL checkout; the detector is the separate
local /api/tag service. No model, agent framework, or MCP import is needed here.
"""

from dataclasses import dataclass
from functools import wraps
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pii_eval import parse_tagged


@dataclass(frozen=True)
class Policy:
    """Application-owned configuration, never an argument supplied by the agent."""

    name: str
    version: str
    entities: tuple

    def __post_init__(self):
        if not all(isinstance(s, str) and re.fullmatch(r"[a-z0-9_-]{1,64}", s)
                   for s in (self.name, self.version)):
            raise ValueError("policy name and version must be short identifiers")
        if not isinstance(self.entities, tuple) or not 1 <= len(self.entities) <= 32:
            raise ValueError("policy requires a tuple of 1 to 32 entity labels")
        for name in self.entities:
            if (not isinstance(name, str) or not name or len(name) > 64
                    or name != name.strip().lower()
                    or any(c in name for c in "<>/")
                    or any(ord(c) < 32 for c in name)):
                raise ValueError("invalid policy entity label")
        if len(set(self.entities)) != len(self.entities):
            raise ValueError("policy entity labels must be unique")


CREDENTIALS = Policy("credential-exposure", "1", ("password", "api key"))
PERSONAL_DATA = Policy("personal-data", "1", ("person name", "email", "phone number"))
POLICIES = {p.name: p for p in (CREDENTIALS, PERSONAL_DATA)}


class InspectionError(Exception):
    """Inspection did not establish a complete, valid annotation."""


class RetrievalError(Exception):
    """A document cannot be read within the configured retrieval boundary."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A local service must not redirect submitted content to another server.
        return None


class LocalSpaniel:
    """Call the local constrained API; bypass proxies and refuse redirects."""

    def __init__(self, endpoint="http://127.0.0.1:8377/api/tag", timeout=120):
        url = urlsplit(endpoint)
        try:
            loopback = ipaddress.ip_address(url.hostname).is_loopback
            port = url.port
        except (ValueError, TypeError):
            loopback, port = False, None
        if (url.scheme != "http" or not loopback or port is None
                or url.username is not None or url.password is not None
                or url.path != "/api/tag" or url.query or url.fragment):
            raise ValueError("use a literal loopback HTTP address and /api/tag")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.endpoint, self.timeout = endpoint, timeout
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def __call__(self, text, entities):
        request = Request(
            self.endpoint,
            data=json.dumps({"text": text, "entities": list(entities),
                             "constrained": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(1_000_001)
            if len(body) > 1_000_000:
                raise InspectionError()
            return json.loads(body)
        except Exception:
            # Exception bodies can contain submitted text. Never surface them.
            raise InspectionError("local inspection failed") from None


def _finding_count(text, entities, result):
    """Verify the annotation independently of the server's fidelity flag."""
    if (not isinstance(result, dict) or result.get("constrained") is not True
            or result.get("faithful") is not True
            or result.get("entities") != list(entities)
            or not isinstance(result.get("tagged"), str)):
        raise InspectionError()
    # The evaluation parser cannot distinguish source markup from added tags.
    # Refuse this ambiguous representation, including an unchanged model output.
    if any(tag in text for name in entities
           for tag in (f"<{name}>", f"</{name}>")):
        raise InspectionError()
    try:
        recovered, spans = parse_tagged(result["tagged"], entities)
    except ValueError:
        raise InspectionError() from None
    if recovered != text or any(
        end - start < 2 or not text[start:end].strip()
        for start, end, _ in spans
    ):
        raise InspectionError()
    return len(spans)


class ContentGate:
    """Withhold all text on a match or failure. Does not redact or stream."""

    def __init__(self, policy=CREDENTIALS, tagger=None, max_chars=8000):
        if not isinstance(policy, Policy):
            raise ValueError("a validated Policy is required")
        if type(max_chars) is not int or not 1 <= max_chars <= 12000:
            raise ValueError("max_chars must be between 1 and 12000")
        self.policy = policy
        self.tagger = tagger if tagger is not None else LocalSpaniel()
        self.max_chars = max_chars

    def _result(self, status, code, reason, content=None):
        return {"status": status, "content": content, "code": code,
                "reason": reason, "policy": self.policy.name,
                "policy_version": self.policy.version, "retryable": False}

    def retrieval_failed(self):
        return self._result("error", "RETRIEVAL_FAILED",
                            "Content could not be retrieved within the configured limits.")

    def inspect(self, text):
        if not isinstance(text, str) or len(text) > self.max_chars or "\x00" in text:
            return self._result("error", "UNSUPPORTED_CONTENT",
                                "Content withheld: unsupported text or inspection limit exceeded.")
        if not text.strip():
            return self._result("allowed", "EMPTY_CONTENT",
                                "No non-whitespace text to inspect.", text)
        try:
            result = self.tagger(text, self.policy.entities)
            count = _finding_count(text, self.policy.entities, result)
        except Exception:
            return self._result("error", "INSPECTION_FAILED",
                                "Content withheld: a complete valid inspection was unavailable.")
        if count:
            return self._result("blocked", "SENSITIVE_CONTENT_DETECTED",
                                "Content withheld by the configured data protection policy.")
        return self._result("allowed", "NO_MATCHES_DETECTED",
                            "No requested entities detected; this is not a guarantee of safety.", text)


def protect_tool(retrieve, gate):
    """Wrap a synchronous, non-streaming, text-returning retrieval function.

    The inner callable must not emit raw content to agent events, logs, traces,
    or callbacks. Register only the returned callable with the agent.
    """
    @wraps(retrieve)
    def protected(*args, **kwargs):
        try:
            text = retrieve(*args, **kwargs)
        except Exception:
            return gate.retrieval_failed()
        return gate.inspect(text)

    return protected


class LocalDocumentReader:
    """Bounded UTF-8 reads inside one directory, without following symlinks.

    Uses POSIX directory file descriptors to prevent path/symlink substitution
    between validation and open. No offsets or partial-read option are exposed.
    """

    SUFFIXES = {".txt", ".md", ".json", ".csv", ".log"}

    def __init__(self, root, max_bytes=32000):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir() or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("provide a document directory and positive byte limit")
        if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
            raise ValueError("this reader requires POSIX directory-descriptor support")
        self.max_bytes = max_bytes

    def __call__(self, document):
        if not isinstance(document, str) or not document or "\x00" in document:
            raise RetrievalError()
        path = Path(document)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.suffix.lower() not in self.SUFFIXES):
            raise RetrievalError()
        directory = None
        descriptor = None
        try:
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for component in path.parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_bytes:
                raise RetrievalError()
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                payload = stream.read(self.max_bytes + 1)
            if len(payload) > self.max_bytes:
                raise RetrievalError()
            text = payload.decode("utf-8")
            if "\x00" in text:
                raise RetrievalError()
            return text
        except (OSError, UnicodeError, ValueError):
            raise RetrievalError() from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if directory is not None:
                os.close(directory)


def make_document_tool(root, policy=CREDENTIALS, tagger=None):
    """Return an agent-facing callable with exactly one argument: document."""
    protected = protect_tool(LocalDocumentReader(root), ContentGate(policy, tagger))

    def read_document(document: str) -> dict:
        """Read an allowed local text document; content is null if withheld."""
        return protected(document)

    return read_document
