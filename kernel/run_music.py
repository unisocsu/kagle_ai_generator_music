"""Generate music on Kaggle using ACE-Step 1.5."""
import base64, json, subprocess, sys
from pathlib import Path
CONFIG=json.loads(base64.b64decode("__CONFIG_B64__").decode("utf-8"))
ROOT=Path("/kaggle/working/ACE-Step-1.5"); OUT=Path("/kaggle/working/music-output"); CKPT=ROOT/"checkpoints"
def run(cmd,cwd=None): print("+"," ".join(map(str,cmd)),flush=True); subprocess.run(cmd,cwd=cwd,check=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True); run(["nvidia-smi"])
 run(["git","clone","--depth","1","https://github.com/ACE-Step/ACE-Step-1.5.git",str(ROOT)])
 run([sys.executable,"-m","pip","install","-e",".","--no-build-isolation"],cwd=ROOT)
 run([sys.executable,"-m","acestep.model_downloader"],cwd=ROOT)
 from huggingface_hub import snapshot_download
 snapshot_download(repo_id="ACE-Step/acestep-v15-xl-turbo",local_dir=str(CKPT/"acestep-v15-xl-turbo"))
 import torch
 from acestep.handler import AceStepHandler
 from acestep.llm_inference import LLMHandler
 from acestep.inference import GenerationParams,GenerationConfig,generate_music
 print("CUDA devices:",torch.cuda.device_count(),flush=True)
 for i in range(torch.cuda.device_count()): print(f"GPU {i}: {torch.cuda.get_device_name(i)} / {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GiB",flush=True)
 dit=AceStepHandler(); llm=LLMHandler()
 dit.initialize_service(project_root=str(ROOT),config_path="acestep-v15-xl-turbo",device="cuda",offload_to_cpu=True)
 llm.initialize(checkpoint_dir=str(CKPT),lm_model_path="acestep-5Hz-lm-1.7B",backend="pt",device="cuda")
 files=[]
 for i in range(CONFIG["num_samples"]):
  seed=CONFIG["seed"]+i if CONFIG["seed"]>=0 else -1
  params=GenerationParams(task_type="text2music",caption=CONFIG["prompt"],duration=float(CONFIG["duration"]),vocal_language=CONFIG.get("vocal_language","en"),instrumental=CONFIG.get("instrumental",False),lyrics=CONFIG.get("lyrics",""),seed=seed,inference_steps=8,shift=3.0,thinking=True,use_cot_caption=True,use_cot_metas=True)
  result=generate_music(dit,llm,params,GenerationConfig(batch_size=1,audio_format="wav",use_random_seed=(seed<0)),save_dir=str(OUT))
  if not result.success: raise RuntimeError(f"ACE-Step generation failed: {result.error}")
  for audio in result.audios:
   src=Path(audio["path"]); dest=OUT/f"music_{i+1:02d}.wav"
   if src.resolve()!=dest.resolve():
    import shutil; shutil.copy2(src,dest)
   files.append(dest.name)
 info={"engine":"ACE-Step 1.5","dit_model":"acestep-v15-xl-turbo (4B)","lm_model":"acestep-5Hz-lm-1.7B","prompt":CONFIG["prompt"],"gpu_count":torch.cuda.device_count(),"gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"files":files}
 (OUT/"run_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
 print("Generation complete.",flush=True)
if __name__=="__main__": main()
