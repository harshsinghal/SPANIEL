"""Source-grounded annotation, validation, and the experiment's policy contract."""
import hashlib
import re

from pii_eval import parse_tagged

SYSTEM = (
    "You tag entities in text. Reproduce the user's text exactly, wrapping each "
    "entity that matches a requested type in XML tags, like <label>entity</label>. "
    "Use only the requested labels. Tag every match. If nothing matches, reproduce "
    "the text unchanged. Never alter, add, or omit any other characters."
)
POLICY = {
    "version": "credential-context-2026-10-21",
    "password": "An actual or purported authentication secret, including natural-language passphrases and revoked secrets. Exclude explicit unfilled placeholders, masked values, and ordinary phrases without authentication context.",
    "api key": "An actual or purported API credential, bearer token or session token. Exclude unfilled placeholders, public identifiers and format descriptions. Token validity is not checked.",
    "private key": "The complete private-key block, including delimiters and padding. Exclude public keys and certificates. Cryptographic validity is not checked.",
    "private compensation": "A pay amount or range tied to an individual's pay, a private negotiation, a nonpublic internal budget or a confidential compensation conversation. Public advertisements and published general market ranges are benign. A range can be sensitive. Missing disclosure context is ambiguous and must not be silently labeled benign.",
    "uncertainty": "Ambiguous cases are a separate review set, not negative training labels. This model has no calibrated abstention mechanism; empty output cannot resolve ambiguity.",
    "evidence": "Extraction targets are non-overlapping source-contiguous spans. Sidecar evidence groups can link spans to contextual cues; the current model does not output those groups.",
    "transcripts": "Text-only simulated transcripts. Preserve ASR-like errors and source spelling exactly; no audio, ASR model, or real transcription corpus was evaluated.",
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(text, spans):
    parts, cursor = [], 0
    for s in sorted(spans, key=lambda x: x["start"]):
        a, b, label = s["start"], s["end"], s["label"]
        if not 0 <= cursor <= a < b <= len(text) or b-a < 2:
            raise ValueError("invalid or overlapping span")
        if not text[a:b].strip():
            raise ValueError("blank span")
        parts.extend((text[cursor:a], f"<{label}>", text[a:b], f"</{label}>"))
        cursor = b
    return "".join(parts) + text[cursor:]


def validate(row):
    text, labels = row["text"], row["entities"]
    if not text or not labels or len(labels) != len(set(labels)):
        raise ValueError("text and unique labels required")
    for label in labels:
        if not re.fullmatch(r"[^<>/\r\n]{1,64}", label):
            raise ValueError("invalid label")
        if f"<{label}>" in text or f"</{label}>" in text:
            raise ValueError("source collides with annotation syntax")
    if any(s["label"] not in labels for s in row["spans"]):
        raise ValueError("unrequested span label")
    target = render(text, row["spans"])
    recovered, parsed = parse_tagged(target, labels)
    expected = [(s["start"], s["end"], s["label"]) for s in sorted(row["spans"], key=lambda s:s["start"])]
    if recovered != text or parsed != expected:
        raise ValueError("target does not exactly recover source and spans")
    if "messages" in row:
        chat=row["messages"]
        if len(chat)!=3 or [m["role"] for m in chat]!=["system","user","assistant"]:
            raise ValueError("expected system/user/assistant chat")
        if chat[1]["content"].split("\n\nText:\n",1)[-1]!=text or chat[2]["content"]!=target:
            raise ValueError("chat disagrees with source-grounded annotation")
    for group in row.get("evidence_groups", []):
        for span in group["context_spans"]:
            if not 0 <= span["start"] < span["end"] <= len(text):
                raise ValueError("invalid contextual evidence offset")
        if any(i < 0 or i >= len(row["spans"]) for i in group["span_indices"]):
            raise ValueError("invalid evidence group index")
    return target


def messages(row):
    target = validate(row)
    user = "Entity types:\n" + "\n".join("- " + s for s in row["entities"]) + "\n\nText:\n" + row["text"]
    return [{"role":"system", "content":SYSTEM}, {"role":"user", "content":user}, {"role":"assistant", "content":target}]
