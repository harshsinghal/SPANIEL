#!/usr/bin/env python3
"""Build OpenAI Batch API job files for the $10 v2 data-improvement budget.

Jobs:
  A  eval-gold adjudication      gpt-5.6-sol    300 scored eval rows
  B  guideline-conformance audit gpt-5.4-nano   nemotron rows w/ occupation+date family
  C  alias expansion             gpt-5.6-luna   all labels x 20 paraphrases
  D  register-gap synthesis      nano + mini    JSON logs / checkbox forms / prose

Each job's cost is estimated (chars/4 ~ tokens) against batch prices and
scope is trimmed to its budget line. Writes batch_jobs/{A,B,C,D}.jsonl and
prints the projected spend. Submission is a separate script.
"""

import json
import random
import re
from pathlib import Path

OUT = Path("batch_jobs")
OUT.mkdir(exist_ok=True)
rng = random.Random(20260718)

PRICE = {  # $/1M tokens (batch): (input, output)
    "gpt-5.6-sol": (2.50, 15.00),
    "gpt-5.6-luna": (0.50, 3.00),
    "gpt-5.4-nano": (0.10, 0.625),
    "gpt-5.4-mini": (0.375, 2.25),
}
BUDGET = {"A": 2.60, "B": 3.00, "C": 0.25, "D": 3.00}

GUIDELINES = """Label rules (v2 draft, derived from observed gold inconsistencies):
- date: a calendar date with NO time component (e.g. 2023-08-15).
- date_time: must contain BOTH a date and a time (e.g. 2023-10-01T10:00:00Z). A bare date is never date_time.
- time: a clock time alone.
- date_of_birth: a date explicitly denoting someone's birth; takes precedence over date.
- occupation: the full job title or role of a person, including compound descriptors (e.g. 'laborer freight stock or material mover'), EXCLUDING department/organization suffixes ('... department'), document titles, honorific prefixes, and bare credentials (MD, PhD).
- Entities are only spans disclosing information about a person or their record. Empty form glyphs, checkboxes, and placeholder symbols are never entities."""


def req(custom_id, model, system, user, max_tokens):
    return {"custom_id": custom_id, "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model,
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                     "max_completion_tokens": max_tokens}}


def est_cost(lines, model, avg_out_tokens):
    inp = sum(len(json.dumps(l["body"]["messages"])) for l in lines) / 4 / 1e6
    out = len(lines) * avg_out_tokens / 1e6
    p = PRICE[model]
    return inp * p[0] + out * p[1]


def job_a():
    system = ("You audit PII span annotations. " + GUIDELINES +
              "\nFor the document and its gold spans, return JSON only: "
              '{"verdicts":[{"i":<idx>,"v":"ok|wrong_label|wrong_boundary|not_entity",'
              '"fix":"<corrected label or span text, if any>"}],'
              '"missed":[{"text":"<span>","label":"<label>"}]} '
              "Judge every gold span; list clear misses only.")
    rows = [json.loads(l) for l in open("sft_data/sft_eval.jsonl", encoding="utf-8")][:300]
    from pii_eval import parse_example, parse_tagged
    lines = []
    for i, r in enumerate(rows):
        text, names = parse_example(r)
        _, spans = parse_tagged(r["messages"][2]["content"], names)
        gold = [{"i": j, "text": text[s:e], "label": n}
                for j, (s, e, n) in enumerate(spans)]
        user = (f"Requested labels: {', '.join(names)}\n\nDocument:\n{text}\n\n"
                f"Gold spans:\n{json.dumps(gold, ensure_ascii=False)}")
        lines.append(req(f"A-{i:03d}", "gpt-5.6-sol", system, user, 700))
    return lines, "gpt-5.6-sol", 350


def job_b():
    system = ("You audit PII annotations against rules. " + GUIDELINES +
              "\nInput is text with inline <label>entity</label> tags. Return JSON only: "
              '{"violations":[{"entity":"<text>","label":"<current>",'
              '"action":"relabel:<new>|extend:<full span>|shrink:<span>|remove"}]} '
              "Empty list if all tags obey the rules.")
    target = re.compile(r"</(occupation|date|date_time|time|date_of_birth)>")
    lines = []
    with open("pii_xml/nemotron_train.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        t = r["text_xml"]
        if not target.search(t) or len(t) > 2600:
            continue
        lines.append(req(f"B-{len(lines):05d}", "gpt-5.4-nano", system, t, 220))
        if len(lines) >= 40000:
            break
    return lines, "gpt-5.4-nano", 90


def job_c():
    aliases = json.load(open("label_aliases.json"))
    system = ("Generate natural-language paraphrases a user might type to request a "
              "PII entity type. Lowercase, no underscores, 1-5 words each, no "
              "duplicates of the given existing names. Return JSON list of 20 strings.")
    lines = []
    for ds in ("gravitee", "nemotron"):
        for slug, existing in aliases[ds].items():
            user = f"Entity type: {slug}\nExisting names: {existing}\nDescription context: PII detection."
            lines.append(req(f"C-{ds}-{slug}", "gpt-5.6-luna", system, user, 400))
    return lines, "gpt-5.6-luna", 250


def job_d():
    nemo = list(json.load(open("label_aliases.json"))["nemotron"].keys())
    families = [
        ("dlog", "gpt-5.4-nano", 1100, 800,
         "Write a realistic JSON or log-file style machine document (session logs, "
         "API audit trails, transaction records) containing PII values. "),
        ("form", "gpt-5.4-mini", 700, 800,
         "Write a form-style document (markdown tables, checklists with empty "
         "checkbox glyphs like □, rating grids) where most fields are EMPTY "
         "placeholders and only a few contain real PII values. "),
        ("prose", "gpt-5.4-mini", 700, 800,
         "Write an encyclopedia/news register paragraph about a company or event "
         "that mentions dates, places and organizations as world facts. Also "
         "include one or two sentences about a specific person with their "
         "personal details. "),
    ]
    lines = []
    for tag, model, count, max_out, brief in families:
        for i in range(count):
            labs = rng.sample(nemo, rng.randint(4, 8))
            names = [l.replace("_", " ") for l in labs]
            user = (brief + "Wrap every entity matching these types in XML tags "
                    f"using exactly these tag names: {', '.join(names)}. "
                    "Tag only real values, never placeholders. Vary length "
                    "(120-350 words), topic, formatting. Output the tagged "
                    f"document only. Seed: {rng.randint(0, 10**9)}")
            lines.append(req(f"D-{tag}-{i:04d}", model,
                             "You generate synthetic PII training documents.",
                             user, max_out))
    return lines, None, 550


def main():
    total = 0.0
    for name, builder in (("A", job_a), ("B", job_b), ("C", job_c), ("D", job_d)):
        lines, model, avg_out = builder()
        if model:  # single-model job: trim to budget
            cost = est_cost(lines, model, avg_out)
            while cost > BUDGET[name] and len(lines) > 10:
                lines = lines[:int(len(lines) * BUDGET[name] / cost * 0.97)]
                cost = est_cost(lines, model, avg_out)
        else:      # mixed-model (D): estimate per model
            cost = sum(est_cost([l for l in lines if l["body"]["model"] == m],
                                m, avg_out) for m in PRICE)
            while cost > BUDGET[name]:
                lines = [l for i, l in enumerate(lines) if i % 10]  # drop 10%
                cost = sum(est_cost([l for l in lines if l["body"]["model"] == m],
                                    m, avg_out) for m in PRICE)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        print(f"job {name}: {len(lines):>6} requests  est ${cost:.2f}")
        total += cost
    print(f"\nprojected total: ${total:.2f}  (budget $10.00, reserve ${10 - total:.2f})")


if __name__ == "__main__":
    main()
