"""Validates workflow inputs and prepares the Kaggle kernel."""
import base64,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BUILD=ROOT/"build"/"kernel"; TEMPLATE=ROOT/"kernel"/"run_music.py"; PLACEHOLDER="__CONFIG_B64__"
def fail(m): print(f"::error::{m}"); sys.exit(1)
def number(n,d,lo,hi,cast):
 raw=os.environ.get(n,"").strip() or str(d)
 try: v=cast(raw)
 except ValueError: fail(f"{n} must be numeric")
 if not lo<=v<=hi: fail(f"{n} must be between {lo} and {hi}")
 return v
def main():
 owner=os.environ.get("KAGGLE_USERNAME","").strip().lower()
 if not owner: fail("KAGGLE_USERNAME is empty.")
 prompt=os.environ.get("IN_PROMPT","").strip(); lyrics=os.environ.get("IN_LYRICS","").strip()
 if not prompt: fail("prompt is required.")
 if len(prompt)>512 or len(lyrics)>4096: fail("prompt max 512 chars; lyrics max 4096 chars.")
 cfg={"prompt":prompt,"lyrics":lyrics,"duration":number("IN_DURATION",30,10,600,float),"num_samples":number("IN_NUM_SAMPLES",1,1,3,int),"seed":number("IN_SEED",-1,-1,2**31-1,int),"vocal_language":os.environ.get("IN_VOCAL_LANGUAGE","en").strip() or "en","instrumental":os.environ.get("IN_INSTRUMENTAL","false").lower()=="true"}
 template=TEMPLATE.read_text(encoding="utf-8")
 if template.count(PLACEHOLDER)!=1: fail("Invalid template placeholder.")
 BUILD.mkdir(parents=True,exist_ok=True)
 encoded=base64.b64encode(json.dumps(cfg,ensure_ascii=False).encode()).decode()
 (BUILD/"run_music.py").write_text(template.replace(PLACEHOLDER,encoded),encoding="utf-8")
 metadata={"id":f"{owner}/kagle-ai-music-generator","title":"kagle ai music generator","code_file":"run_music.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"true","enable_tpu":"false","enable_internet":"true","dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
 (BUILD/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
 (ROOT/"build"/"kernel_ref.txt").write_text(metadata["id"],encoding="utf-8")
 print(f"Kernel: {metadata['id']}"); print("ACE-Step 1.5 XL Turbo 4B DiT + 1.7B LM")
if __name__=="__main__": main()
