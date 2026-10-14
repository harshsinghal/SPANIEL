# Governed retrieval: inspect locally before returning tool content

This Python example connects SPANIEL's sensitive-entity detection to a concrete
AI governance control. A retrieval tool reads a document locally, inspects the
captured text, and returns either the original content or a denial with
`content: null`. The agent does not receive the intermediate document.

Implemented in [spaniel_guard.py](../spaniel_guard.py), with
[synthetic example documents](../examples/governed_retrieval/documents), a
[live evaluation runner](../examples/governed_retrieval/demo.py), and
[enforcement tests](../tests/test_spaniel_guard.py). The core wrapper uses only
the Python standard library and the repository's annotation parser. It calls
the separate local SPANIEL service; it does not load model weights itself.

```text
Agent calls read_document(document)
    -> bounded local read, held in private memory
    -> local SPANIEL annotation under a configured policy
    -> validate annotation and apply deterministic release rule
         requested entity found -> blocked; content = null
         inspection failed      -> error;   content = null
         no entity found        -> allowed; return the captured original
```

## Run the example

Use Python 3.9+ for the wrapper and tests. The bundled file reader uses POSIX
directory descriptors and runs on macOS/Linux. The model service has its own
dependencies; start it using the [README instructions](../README.md#try-it-the-demo-app).
Run the following from the repository root with the service on port 8377:

```bash
python3 examples/governed_retrieval/demo.py benign_public.txt
python3 examples/governed_retrieval/demo.py sensitive_password.txt
python3 examples/governed_retrieval/demo.py does-not-exist.txt
```

A detected credential produces this result. It is an illustrative response,
not a promise that every credential will be detected:

```json
{
  "status": "blocked",
  "content": null,
  "code": "SENSITIVE_CONTENT_DETECTED",
  "reason": "Content withheld by the configured data protection policy.",
  "policy": "credential-exposure",
  "policy_version": "1",
  "retryable": false
}
```

Allowed results have `status: "allowed"`, `code: "NO_MATCHES_DETECTED"`, and the
original text in `content`. Empty/whitespace documents have `EMPTY_CONTENT` and
skip inference. No finding is a policy-scoped result, not a declaration that
the document is safe. Single reads exit 0 for allowed and 2 for withheld results.

Error codes distinguish `RETRIEVAL_FAILED` (missing/unreadable/disallowed file),
`UNSUPPORTED_CONTENT` (unsupported text or length), and `INSPECTION_FAILED`
(timeout, service error, malformed annotation, copy drift, or ambiguous tags).
Errors always have null content. Detailed exceptions are not returned: they
can contain paths, response bodies, or sensitive text.

## Bind policy before registering the tool

```python
from spaniel_guard import CREDENTIALS, make_document_tool

read_document = make_document_tool(
    root="examples/governed_retrieval/documents",
    policy=CREDENTIALS,
)
result = read_document("sensitive_password.txt")
```

The callable's only argument is `document: str`. Its policy, document directory,
and detector endpoint are application configuration, not model-controlled tool
arguments. There is no bypass, threshold, alternate endpoint, or partial-read
argument. The policy configuration is frozen; policy edits require application
configuration changes. The `retryable` field communicates intent to a consumer;
repeated or alternative calls still need enforced controls.

Two example policies are provided:

| Policy | Requested entities | Release rule |
|---|---|---|
| `credential-exposure`, version `1` | `password`, `api key` | Withhold the whole result on any finding |
| `personal-data`, version `1` | `person name`, `email`, `phone number` | Withhold the whole result on any finding |

These are deliberately narrow demonstration policies. A personal-data policy
that blocks all names can overblock public information. Compensation and legal
confidentiality need additional detection and policy evaluation; adding an
entity label to `Policy` does not validate the model's ability to recognize it.

The default local HTTP client accepts only literal loopback addresses, such as
`http://127.0.0.1:8377/api/tag`. It disables environment proxies and refuses HTTP
redirects. It requires `constrained: true`, `faithful: true`, the expected label
list, well-formed tags, and an independently verified exact copy of the source.
The tag check is deterministic: a parser pairs opening/closing tags for the
requested labels and records their character offsets. Removing those annotation
tags must recover the entire original string exactly. Each finding must enclose
at least two original characters and contain a non-whitespace character. Empty,
whitespace-only, newly introduced self-closing, nested, or unbalanced tags and
invented text cause inspection failure with null content. Literal requested tags
already present in the source also cause failure because they make this
representation ambiguous. These are XML-like annotations, not general XML:
labels such as `api key` contain spaces, so the checker uses the project's tag
grammar rather than an XML document parser. A structurally valid output with no
findings can still be a model false negative.
There is no fallback to unconstrained generation or a hosted detector. Trust
and provision the local service itself: this protocol does not authenticate or
attest its model version.

## Wrap another retriever or search implementation

```python
from spaniel_guard import ContentGate, CREDENTIALS, protect_tool

def retrieve_text(document_id: str) -> str:
    # Replace this body with your trusted local retrieval implementation.
    # Return text here; do not emit it to the agent or a tracing service.
    return "Public release notes: the search feature is now available."

guarded_retrieve = protect_tool(retrieve_text, ContentGate(CREDENTIALS))
result = guarded_retrieve("release-notes")
```

`protect_tool` supports synchronous, non-streaming functions returning one
string. For search, explicitly serialize **every field you intend to return**
(titles, snippets, URLs, metadata) to that string inside the private retriever.
The wrapper withholds the entire result if any requested entity is detected.
It does not silently scan one field of a dictionary while returning the others;
non-string results are rejected. Structured per-result filtering is future work.

Place the wrapper before the framework records a tool result. Calling an already
registered raw tool and then inspecting its visible result is too late. The
inner retriever must not stream raw updates, call an external summarizer, or
send content to logs/traces. The wrapper cannot undo disclosures already made
by the underlying retrieval implementation or search backend.

## Agent registration and MCP exposure

For a Python agent framework, register the `read_document` callable from
`make_document_tool` using that framework's function-tool adapter. Publish an
input schema with one required string, `document`, and no additional properties.
The adapter must preserve the returned object, including blocked/error status
and null content. Register only the guarded callable; do not also expose the
raw reader. Avoid supplying the original retriever's return annotation (`str`)
as the wrapper's output schema: the guarded result is a dictionary.

For MCP, the same callable can be exposed by a small local MCP server. For
example, the official Python SDK v2 uses `MCPServer` and `@mcp.tool()`:

```python
from mcp.server import MCPServer
from spaniel_guard import make_document_tool

mcp = MCPServer("Spaniel governed retrieval")
guarded_read = make_document_tool("examples/governed_retrieval/documents")

@mcp.tool()
def read_document(document: str) -> dict:
    """Read an allowed document; content is null if policy withholds it."""
    return guarded_read(document)
```

This is an integration recipe, not an installed MCP dependency or a tested
end-to-end agent integration. Use the SDK's supported launcher/transport and
keep its process in the trusted local environment. See the
[official Python SDK guide](https://github.com/modelcontextprotocol/python-sdk)
for installation and launch commands. MCP clients may render structured and
text content separately: neither should receive the original when denied.
Return a valid MCP tool-result envelope containing the application decision,
not a bare JSON-RPC null. An expected policy denial can be represented as a
normal structured result with `status: "blocked"`; it is not an empty search.

For Pi, a Python-backed tool or bridge can invoke this callable, or an extension
can override an existing tool name. The
[official tool override example](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/tool-override.ts)
shows that registration mechanism. This repository does not install or change
any agent's configuration.

## Examples, failures, and accuracy

All bundled documents are newly authored synthetic fixtures. Credentials are
invented, emails use `example.com`, and the phone number is fictional. They are
not copied from training datasets or private documents. Treat the synthetic
values as real sensitive values when evaluating the expected actions.

The [case manifest](../examples/governed_retrieval/cases.json) records the selected
policy and an author-defined expected outcome for each document:

| Group | Purpose |
|---|---|
| Benign | Public prose and security guidance without credential values |
| Sensitive | Explicit password, API key, personal details, Unicode password |
| Ambiguous | Implicit passphrase, template placeholder, revoked credential |
| Scope | Personal data, compensation, and legal strategy under a credential-only policy |
| Adversarial | A document instruction requesting that detection be skipped |
| Error | Missing document and literal markup that collides with annotation tags |

The credential policy's intended semantics include revoked credential values
and implicit login passphrases, but exclude obvious unfilled placeholders. These
expectations are reviewable test labels, not automatically learned guarantees.
Scope cases intentionally expect release under a credential-only policy; their
content can still be confidential under a different policy.

Run actual model inference over every fixture:

```bash
python3 examples/governed_retrieval/demo.py --suite --report /tmp/spaniel-retrieval-results.json
```

The runner prints expected versus actual actions, content-release status, and
elapsed time. Its report contains no document text or detected values. Exit 1
means at least one case differed from its expected action. It separately counts
**sensitive content released**, **false blocks**, and **unexpected errors**; an
inspection error on a sensitive example does not count as successful detection.
These are small synthetic document-level smoke cases, not a held-out benchmark
or span F1. The model endpoint does not report its revision, so record the server
configuration and hardware alongside any published results.

### Recorded local run

The [October 13, 2026 report](../examples/governed_retrieval/results/local-smoke-2026-10-13.json)
records one real run of the published v2 checkpoint at revision
`a58063469be8c7b929016d944d8d6d1545bc70a0` on an Apple M4 with 16 GiB memory,
using PyTorch MPS. **13 of 15 cases matched their expected actions**, including
the two expected error cases. No unexpected inspection errors occurred.

| Observed outcome | Detail |
|---|---|
| Sensitive content released | The implicit login passphrase was not tagged and was returned |
| False block | `YOUR_PASSWORD_HERE` in a documentation template was tagged and withheld |
| Explicit sensitive cases | The password, API key, personal-data, and Unicode cases were withheld |
| Document instruction | The credential was detected despite an instruction to leave the text untagged |

These two accuracy failures are intentionally preserved. They are useful
targets for future model evaluation, not reasons to change expected labels to
match predictions. The successful instruction case does not establish general
prompt-injection resistance. Client timings for the 14 model calls ranged from
0.430 to 2.507 seconds, including first-call warm-up; these are tiny documents
in one sequential run, not a throughput or long-document latency benchmark.
The report includes policy names, outcomes, launch configuration, and fixture
hashes without raw document content.

There is no calibrated confidence score or low-confidence branch in the current
API. Ambiguous cases reveal actual false positives/negatives; the tool does not
invent an uncertainty estimate from copy fidelity or the absence of tags.

Run controlled enforcement tests without a model or network service:

```bash
python3 -m unittest tests.test_spaniel_guard -v
make test
```

These tests inject detector outputs and failures. They cover withholding,
timeouts, invalid JSON, HTTP errors/redirects, incomplete or malformed tags,
copy drift, oversized inputs, invalid UTF-8, path traversal, symlinks, special
files, and exact source preservation. They also explicitly demonstrate a
simulated false negative releasing a secret and a simulated false positive
blocking a placeholder. They are not evidence of model accuracy.

## Boundary and limits

- **Text only:** UTF-8 `.txt`, `.md`, `.json`, `.csv`, and `.log`, at most 32,000
  bytes and 8,000 characters by default. No OCR, archives, images, remote URL
  fetching, chunking, or silent truncation. Oversized documents are withheld.
- **Same captured text:** the reader captures the document once; the gate
  releases that exact string if allowed. It does not reopen the file after
  inspection. Concurrent edits can affect what a read captures, but everything
  in the captured string is inspected before release.
- **Local path boundary:** relative paths within an administrator-selected
  directory, regular files only, no symlink traversal. Keep this directory and
  the policy/service configuration under trusted administration.
- **Other access paths:** shell commands, another reader, direct MCP resources,
  attachments, and existing conversation content can bypass this particular
  tool. Restrict or protect those routes before claiming agent-wide coverage.
- **No raw audit log:** the wrapper logs nothing. The caller can record decision
  codes, policy versions, and timings; do not log full returned objects when
  allowed content may itself be sensitive outside the selected policy.
- **Learning remains imperfect:** valid constrained annotation proves source
  preservation, not semantic recall. A missed detection can release sensitive
  content. Entity detection also does not establish legal privilege, business
  confidentiality, or authorization to disclose.

The testable claim is: **when this gate rejects a retrieval result, it returns
no document content through that tool response.** Broader prevention claims
require deployment coverage and representative detection evaluations.
