"""Build synthetic contrast pairs and disjoint scenario-family splits. No network calls."""
import argparse
import base64
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from .schema import POLICY, messages, sha, validate
from .scenarios import SCENARIOS, AMBIGUOUS

SEED = 20261021
WORDS = {
    "train": ("amber cedar coral willow silver hazel meadow pebble velvet maple".split(), "river trail cloud garden lantern valley breeze forest feather horizon".split()),
    "dev": ("cobalt copper indigo mossy violet".split(), "harbor canyon bridge summit orchard".split()),
    "test": ("saffron scarlet ivory azure crimson".split(), "glacier tundra fountain prairie lagoon".split()),
}
NAMES = {"train":["Avery","Samira","Devon","Elena","Rowan"], "dev":["Imani","Oscar"], "test":["Nadia","Felix","Priya"]}


def values(rng, split, category, label, i):
    words = WORDS[split]
    phrase = " ".join([rng.choice(words[0]), rng.choice(words[1]), rng.choice(words[1])])
    opaque = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789") for _ in range(rng.choice([18,24,32,48])))
    secret = phrase if "passphrase" in category or category == "transcript_credential" else phrase.replace(" ", "_") + f"!{rng.randrange(10,99)}"
    if label == "api key":
        secret = rng.choice(["audit_", "sk_demo_", "ghp_", "xoxb_", ""]) + opaque
        if category == "access_token":
            payload = base64.urlsafe_b64encode(json.dumps({"sub":"fictional-"+opaque[:12],"jti":opaque[12:]}).encode()).decode().rstrip("=")
            secret = "eyJhbGciOiJIUzI1NiJ9."+payload+"."+opaque
    if category == "noisy_credential" and label == "api key":
        secret = secret[:len(secret)//2]+"\n"+secret[len(secret)//2:]
    keybody = base64.b64encode(("NONOPERATIONAL-SYNTHETIC-"+opaque).encode()).decode()
    public_key = "-----BEGIN PUBLIC KEY-----\n"+keybody+"\n-----END PUBLIC KEY-----"
    if label == "private key":
        secret = "-----BEGIN PRIVATE KEY-----\n"+keybody+"\n-----END PRIVATE KEY-----"
    lower = rng.randrange({"train":70,"dev":75,"test":85}[split],190,10)
    upper = lower + rng.choice([10,20,30])
    pay = rng.choice([f"${lower},000–${upper},000",f"${lower}k to ${upper}k",f"USD {lower},000 to {upper},000"])
    tens = {0:"",10:"ten",20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
    def informal(n):
        tail = n % 100
        word = "fifteen" if tail==15 else ((tens[tail//10*10]+(" five" if tail%10==5 else "")).strip())
        if n < 100: return word
        return ("one " if n<200 else "two ")+(word or "hundred")
    spoken_pay = rng.choice([f"{lower} to {upper} thousand", f"{lower}k maybe {upper}k",f"{informal(lower)} to {informal(upper)}",f"{informal(lower)} ish maybe {informal(upper)}"])
    if "personal_compensation" == category:
        pay = rng.choice([f"${lower},000",f"USD {lower},000",f"${lower}k"])
    return dict(secret=secret, public_key=public_key, name=rng.choice(NAMES[split]), pay=pay, spoken_pay=spoken_pay,
                placeholder=rng.choice(["YOUR_API_KEY_HERE","REPLACE_ME","<token>","${API_KEY}"]) if label=="api key" else rng.choice(["YOUR_PASSWORD_HERE","CHANGEME","[REDACTED]","********"]))


def interpolate(template, fields):
    return re.sub(r"\{(secret|public_key|name|pay|spoken_pay|placeholder)\}", lambda m:fields[m[1]], template)


def annotated(template, fields, label):
    marked = interpolate(template, fields)
    text, spans, cursor = "", [], 0
    for match in re.finditer(r"<<([\s\S]*?)>>", marked):
        text += marked[cursor:match.start()]
        start = len(text); text += match[1]
        spans.append({"start":start,"end":len(text),"label":label})
        cursor = match.end()
    text += marked[cursor:]
    return text, spans


def make_row(split, family, category, label, template, fields, variant, polarity):
    text, spans = annotated(template, fields, label)
    if bool(spans) != (polarity == "sensitive"):
        raise ValueError("polarity/annotation mismatch")
    row = dict(id=f"{family}-{variant:04d}-{polarity}", family=family, pair_id=f"{family}-{variant:04d}",
               split=split, category=category, text=text, entities=[label], spans=spans, decision=polarity,
               policy_version=POLICY["version"], provenance="Codex-authored template and deterministic fictional values; no independent human gold review")
    row["sha256"] = sha(text)
    row["evidence_groups"] = []
    if spans:
        context = []
        if spans[0]["start"]: context.append(dict(start=0,end=spans[0]["start"]))
        if spans[-1]["end"] < len(text): context.append(dict(start=spans[-1]["end"],end=len(text)))
        row["evidence_groups"] = [dict(span_indices=list(range(len(spans))),context_spans=context,
                                       note="Surrounding context, not a minimal or independently adjudicated rationale; not a model target")]
    row["messages"] = messages(row)
    return row


def add_dilution(row, words, position, rng):
    subjects = ["The public timetable","The visitor guide","The museum brochure","The published route map","The festival program","The public library notice"]
    verbs = ["describes the walking route", "lists the opening hours", "explains the seating plan", "shows the exhibition schedule", "includes directions to the entrance", "describes the available facilities"]
    tails = ["for visitors arriving this week.","with details on the publicly accessible website.","along with the printed guide at the entrance.","for anyone planning a visit."]
    paragraphs = []
    while sum(len(x.split()) for x in paragraphs) < words:
        paragraphs.append(rng.choice(subjects)+" "+rng.choice(verbs)+" "+rng.choice(tails))
    cut = round(len(paragraphs)*position)
    prefix = "\n".join(paragraphs[:cut])+"\n\n" if cut else ""
    suffix = "\n\n"+"\n".join(paragraphs[cut:]) if cut<len(paragraphs) else ""
    out = dict(row)
    out.pop("messages",None)
    out["text"] = prefix+row["text"]+suffix
    out["spans"] = [{**s,"start":s["start"]+len(prefix),"end":s["end"]+len(prefix)} for s in row["spans"]]
    out["evidence_groups"] = [{**g,"context_spans":[{k:v+len(prefix) for k,v in s.items()} for s in g["context_spans"]]} for g in row["evidence_groups"]]
    out["id"] += f"-long-{words}-{position}"
    out["pair_id"] += f"-long-{words}-{position}"
    out["category"] = "long_"+row["category"]
    out["dilution"] = dict(target_filler_words=words,evidence_position=position,synthetic_filler=True)
    out["sha256"] = sha(out["text"])
    out["messages"] = messages(out)
    return out


def build(variants=100):
    rows = defaultdict(list); seen = {}; families = {}
    for split,family,category,label,positive,negative in SCENARIOS:
        if family in families: raise ValueError("duplicate scenario family")
        families[family] = split
        count = variants if split=="train" else 3
        for i in range(count):
            rng = random.Random(f"{SEED}:{split}:{family}:{i}")
            fields = values(rng,split,category,label,i)
            for polarity, template in [("sensitive",positive),("benign",negative)]:
                row = make_row(split,family,category,label,template,fields,i,polarity)
                digest = row["sha256"]
                if digest in seen:
                    if seen[digest] != (split,polarity,label): raise ValueError("conflicting duplicate or cross-split leak")
                    continue
                seen[digest] = (split,polarity,label)
                rows[split].append(row)
                if i < (4 if split=="train" else 1) and category in ("passphrase","api_key","compensation_range"):
                    for words in (250,900):
                        for position in (0.0,0.5,1.0):
                            long = add_dilution(row,words,position,random.Random(row["id"]+str(words)))
                            if long["sha256"] in seen: raise ValueError("duplicate long example")
                            seen[long["sha256"]] = (split,polarity,label)
                            rows[split].append(long)
    for family,label,text in AMBIGUOUS:
        rows["review"].append(dict(id=family,family=family,split="review",text=text,entities=[label],spans=[],
                                   decision="review_required",sha256=sha(text),policy_version=POLICY["version"],
                                   note="No asserted negative gold. Do not use in SFT or binary accuracy calculations."))
    return dict(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True);ap.add_argument("--variants",type=int,default=100)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=build(args.variants)
    manifest={"seed":SEED,"policy_version":POLICY["version"],"variants_per_train_template":args.variants,"splits":{},
              "scope":"Synthetic authored scenario-family splits. Substitutions are not independent documents; no human gold or real audio validation.",
              "split_rule":"All substitutions, positive/negative counterfactuals and long variants of a scenario family stay in the assigned split. Dev/test values use separate pools.",
              "generator_files":{p.name:sha(p.read_text()) for p in [Path(__file__),Path(__file__).with_name("scenarios.py"),Path(__file__).with_name("schema.py")]}}
    for split,items in rows.items():
        for row in items:
            if split!="review":validate(row)
        path=args.out/f"{split}.jsonl";path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
        manifest["splits"][split]={"rows":len(items),"families":len({r['family'] for r in items}),"decisions":dict(Counter(r['decision'] for r in items)),
                                   "categories":dict(Counter(r.get('category','ambiguous') for r in items)),"sha256":sha(path.read_text())}
    (args.out/"policy.json").write_text(json.dumps(POLICY,indent=2)+"\n")
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
