#!/usr/bin/env python3
import json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"notebooks"
RES=ROOT/"results"
OUT=NB/"Brinjal_Final_Preprocessed_reconstructed.ipynb"

sources={}
pat=re.compile(r"^# %% \[Cell (\d+)\]\n",re.M)
for p in sorted(NB.glob("source_part_*.py")):
    t=p.read_text(encoding="utf-8")
    ms=list(pat.finditer(t))
    for j,m in enumerate(ms):
        cid=int(m.group(1))
        sources[cid]=t[m.end():ms[j+1].start() if j+1<len(ms) else len(t)].rstrip()+"\n"

outputs={}
for p in sorted(RES.glob("execution_output_part_*.json")):
    for e in json.loads(p.read_text(encoding="utf-8")):
        outputs.setdefault(int(e["cell"]),[]).append(e["output"])

cells=[]
for cid in sorted(set(sources)|set(outputs)):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":outputs.get(cid,[]),"source":sources.get(cid,"")})

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Created {OUT} with {len(cells)} cells")
