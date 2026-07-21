#!/usr/bin/env python3
"""Span-F1 reward for GRPO training of the tagged-regeneration task.

The reward for a completion is the relaxed (overlap) span-level F1 of its
final answer (the text after any </think>) against the gold spans, computed
per the request's label names. A completion whose answer is malformed or
alters the source text scores 0 — the same honest accounting as the eval.

Used as a TRL GRPOTrainer reward function; also self-tests standalone.
"""

import re


def _parse_tags(text, names):
    """Recover (start, end, name) spans from tagged text, in plain-text space.

    Returns (spans, plain_text). Never raises: malformed structure just stops
    parsing, which yields a low reward rather than a crash.
    """
    if not names:
        return [], text
    token = re.compile(
        "<(/?)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        + ")>")
    out, spans, pos, length = [], [], 0, 0
    open_name, open_at = None, None
    for m in token.finditer(text):
        chunk = text[pos:m.start()]
        out.append(chunk)
        length += len(chunk)
        closing, name = m.group(1) == "/", m.group(2)
        if not closing:
            if open_name is not None:
                return spans, "".join(out) + text[pos:]   # nested → bail
            open_name, open_at = name, length
        else:
            if open_name != name:
                return spans, "".join(out) + text[pos:]    # mismatch → bail
            spans.append((open_at, length, name))
            open_name = None
        pos = m.end()
    out.append(text[pos:])
    return spans, "".join(out)


def _f1(gold, pred, relaxed=True):
    unmatched = list(pred)
    tp = 0
    for g in gold:
        for p in unmatched:
            if g[2] != p[2]:
                continue
            hit = (p[0] < g[1] and g[0] < p[1]) if relaxed else (g[:2] == p[:2])
            if hit:
                unmatched.remove(p)
                tp += 1
                break
    fp, fn = len(unmatched), len(gold) - tp
    prec = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def score_one(completion, gold_tagged, names, source_text):
    """Reward in [0,1] for a single completion string."""
    answer = re.sub(r"^[\s\S]*?</think>\s*", "", completion, count=1) \
        if "</think>" in completion else completion
    gold_spans, gold_plain = _parse_tags(gold_tagged, names)
    pred_spans, pred_plain = _parse_tags(answer, names)
    if pred_plain != source_text:      # drift / malformed → forfeit
        return 0.0
    return _f1(gold_spans, pred_spans, relaxed=True)


def make_reward_fn():
    """TRL GRPO reward function: (prompts, completions, **cols) -> list[float].

    Expects the dataset to carry gold_tagged / names / source_text columns,
    passed through as keyword lists aligned with completions.
    """
    def reward(completions, gold_tagged, names, source_text, **kwargs):
        out = []
        for c, g, n, s in zip(completions, gold_tagged, names, source_text):
            text = c[0]["content"] if isinstance(c, list) else c
            out.append(score_one(text, g, n, s))
        return out
    return reward


if __name__ == "__main__":
    src = "Reach Anita at anita.k@corp.io."
    gold = "Reach <person name>Anita</person name> at <email>anita.k@corp.io</email>."
    names = ["person name", "email"]
    tests = [
        ("<think>reasoning</think>\n\n" + gold, 1.0),          # perfect
        (gold, 1.0),                                            # no think, perfect
        ("Reach <person name>Anita</person name> at anita.k@corp.io.", None),  # partial
        ("Reach Anita at anita.k@corp.io.", 0.0),               # nothing tagged
        ("Reach <person name>Anita</person name> at anita.k@corp.iXX.", 0.0),  # drift
    ]
    for comp, expect in tests:
        r = score_one(comp, gold, names, src)
        ok = "OK" if (expect is None or abs(r - expect) < 1e-6) else "FAIL"
        print(f"{ok}  reward={r:.3f}  {comp[:55]!r}")
