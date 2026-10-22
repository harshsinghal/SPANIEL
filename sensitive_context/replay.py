"""Validate a legacy SFT mixture and sample only public-source training replay.

Never edits the original mixture. Rejected rows are referenced by index/hash,
without copying their text into the report. No evaluation data is sampled.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from pii_eval import parse_tagged
from .schema import sha, validate


def convert(row, index):
    head,text=row["messages"][1]["content"].split("\n\nText:\n",1)
    names=[x[2:].split(": ",1)[0] for x in head.splitlines()[1:] if x.startswith("- ")]
    recovered,spans=parse_tagged(row["messages"][-1]["content"],names)
    if recovered!=text:raise ValueError("source recovery mismatch")
    result={"id":f"legacy-{index}","family":f"legacy-{sha(text)}","split":"train", "category":"legacy_"+row["source"],
            "text":text,"entities":names,"spans":[dict(start=a,end=b,label=n) for a,b,n in spans],
            "decision":"sensitive" if spans else "benign", "source":row["source"],"training_row":index,
            "sha256":sha(text),"messages":row["messages"]}
    validate(result)
    return result


def sample(path, per_source=512):
    pools={s:[] for s in ("nemotron","gravitee","ai4privacy")}
    rng={s:random.Random("20261021:"+s) for s in pools}
    seen=Counter();rejected=[];hasher=hashlib.sha256();total=0
    with Path(path).open("rb") as f:
        for i,line in enumerate(f):
            total+=1;hasher.update(line);row=json.loads(line)
            try: record=convert(row,i)
            except (ValueError,KeyError,IndexError) as e:
                rejected.append(dict(training_row=i,source=row.get("source"),row_sha256=hashlib.sha256(line).hexdigest(),issue=str(e)))
                continue
            source=row["source"]
            if source not in pools:continue
            seen[source]+=1;pool=pools[source]
            if len(pool)<per_source:pool.append(record)
            else:
                j=rng[source].randrange(seen[source])
                if j<per_source:pool[j]=record
    items=[r for s in pools for r in pools[s]]
    hashes=[r["sha256"] for r in items]
    if len(hashes)!=len(set(hashes)):raise ValueError("duplicate replay text; adjust sampling and inspect")
    report={"source_sha256":hasher.hexdigest(),"source_rows":total,"valid_public_source_rows":dict(seen),
            "sampled_rows":len(items),"sampled_per_source":{s:len(p) for s,p in pools.items()},"rejected_rows":rejected,
            "scope":"Full legacy target validation; seeded reservoir sample. Private fourth-source examples excluded from replay export. No source artifact changed."}
    return items,report


def main():
    p=argparse.ArgumentParser();p.add_argument("--legacy-train",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--per-source",type=int,default=512)
    a=p.parse_args()
    if a.per_source<1:raise ValueError("per-source must be positive")
    rows,report=sample(a.legacy_train,a.per_source);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    report["replay_sha256"]=sha(a.out.read_text())
    a.out.with_suffix(".validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="rejected_rows"},indent=2));print("Rejected",len(report["rejected_rows"]),"rows")


if __name__=="__main__": main()
