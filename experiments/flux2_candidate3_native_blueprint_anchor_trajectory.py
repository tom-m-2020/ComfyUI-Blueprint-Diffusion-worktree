"""Phase 20c: persistent Blueprint/H trajectory with exact prediction anchoring."""
from __future__ import annotations
import gc, hashlib, json, math, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]; COMFY_ROOT=Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0,str(COMFY_ROOT)); sys.path.insert(0,str(ROOT/"experiments"))
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_blueprint_prediction_anchor as phase20b
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
from blueprint_diffusion.sampling.euler import BlueprintCoordinator,validate_schedule

OUTPUT=ROOT/"experiments"/"flux2_candidate3_native_blueprint_anchor_trajectory_results"
INTERVALS=OUTPUT/"persistent_blueprint_intervals"; REPORT=OUTPUT/"persistent_blueprint_report.json"

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8") as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def stable_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def paths(i): return INTERVALS/f"interval_{i}.json",INTERVALS/f"interval_{i}.pt"
def rms(x): return float(x.detach().float().square().mean().sqrt())
def load_interval(i,key):
    jp,tp=paths(i)
    if not jp.is_file() or not tp.is_file(): return None
    m=json.loads(jp.read_text(encoding="utf-8")); t=torch.load(tp,map_location="cpu",weights_only=True)
    if not m.get("complete") or m.get("configuration_hash")!=key or t.get("configuration_hash")!=key: raise RuntimeError(f"Phase 20c artifact mismatch at {i}")
    return m["record"],t
def save_interval(i,key,record,tensors):
    jp,tp=paths(i); atomic_torch(tp,{"configuration_hash":key,**tensors}); atomic_json(jp,{"complete":True,"configuration_hash":key,"record":record})

class Phase20cSampler(phase14.Phase14Sampler):
  def sample(self,model,sigmas,extra_args,callback,noise,latent_image=None,denoise_mask=None,disable_pbar=False):
    if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)): raise ValueError("Phase 20c requires empty-latent T2I")
    validate_schedule(sigmas)
    if len(sigmas)!=5: raise ValueError("Phase 20c requires the existing four intervals")
    sampling=model.inner_model.model_sampling
    h=sampling.noise_scaling(sigmas[0],noise,latent_image,self.max_denoise(model,sigmas))
    blueprint,bp_coarse,bp_noise=phase20.make_blueprint_state(h,sigmas[0])
    coordinator=BlueprintCoordinator(); coordinator.initialize(h,sigmas[0])
    regions=phase9b.DestinationPlanner().plan(phase14.H_HW)
    if len(regions)!=55: raise AssertionError("Phase 20c requires 55 regions")
    config={"phase":"20c-persistent-blueprint","H":list(phase14.H_HW),"blueprint":list(phase20.BLUEPRINT_HW),"blueprint_tokens":phase20.BLUEPRINT_TOKENS,"regions":55,"seed":phase14.SEED,"sigmas":[float(x) for x in sigmas],"H0_hash":phase14.tensor_hash(h),"blueprint0_hash":phase14.tensor_hash(blueprint),"blueprint0":"Phase-20 construction once","blueprint_coordinates":{"frame":"ordinary_native","y":[0,31],"x":[0,63]},"W":"ordinary normalized W from accepted H crop","D_B":"2x2 mean","U_B":"2x nearest","anchor_strength":1.0,"accepted_states":["blueprint","H"]}
    key=stable_hash(config); base=extra_args["model_options"]; records=[]; outputs=[]
    for ordinal in range(4):
      sigma,sigma_next=sigmas[ordinal],sigmas[ordinal+1]; loaded=load_interval(ordinal,key)
      if loaded:
        rec,t=loaded
        if phase14.tensor_hash(h)!=rec["accepted_input_H_hash"] or phase14.tensor_hash(blueprint)!=rec["accepted_input_blueprint_hash"]: raise RuntimeError("resume lineage mismatch")
        h=t["accepted_H"].to(h.device,h.dtype); blueprint=t["accepted_blueprint"].to(blueprint.device,blueprint.dtype)
        records.append(rec); outputs.append({k:t[k] for k in ("blueprint_x0","ordinary_x0_H","anchored_x0_H","accepted_H")}); print(f"phase20c interval {ordinal} resume-skip {ordinal+1}/4",flush=True); continue
      print(f"phase20c interval {ordinal} start {ordinal+1}/4",flush=True)
      h_hash=phase14.tensor_hash(h); bp_hash=phase14.tensor_hash(blueprint); gc.collect(); phase2.comfy.model_management.soft_empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); wall=time.perf_counter()
      a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); bp_x0=model(blueprint,sigma.expand(1),model_options=phase20.phase8i_options(base),seed=phase14.SEED); b.record(); torch.cuda.synchronize(); bp_ms=float(a.elapsed_time(b))
      mapped=F.interpolate(bp_x0.float(),size=phase14.H_HW,mode="bilinear",align_corners=False).to(bp_x0.dtype)
      working=[]; bp_crops=[]
      for region in regions:
        working.append(phase9c.make_working(h[:,:,region.y:region.y2,region.x:region.x2],sigma,ordinal,region)); bp_crops.append(mapped[:,:,region.y:region.y2,region.x:region.x2])
      w_hashes=[phase14.tensor_hash(x) for x in working]
      a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); x0_w=[model(x,sigma.expand(1),model_options=phase20.phase8i_options(base),seed=phase14.SEED) for x in working]; b.record(); torch.cuda.synchronize(); local_ms=float(a.elapsed_time(b))
      ordinary=[]; anchored=[]; before=[]; after=[]; ratios=[]
      for pred,crop in zip(x0_w,bp_crops):
        fixed,corr=phase20b.correct_prediction(pred,crop); ordinary.append(phase20b.d_blueprint(pred)); anchored.append(phase20b.d_blueprint(fixed)); before.append(rms(ordinary[-1].float()-crop.float())); after.append(float((anchored[-1].float()-crop.float()).abs().max())); ratios.append(rms(corr)/rms(pred))
      if max(after)>1e-5: raise RuntimeError(f"anchor tolerance failed {max(after)}")
      ordinary_h,cov_o=coordinator.assembler.assemble(ordinary,regions,phase14.H_HW); anchored_h,cov_a=coordinator.assembler.assemble(anchored,regions,phase14.H_HW)
      overlap_o=phase8d.overlap_metrics([x.detach().float().cpu() for x in ordinary],regions)["aggregate_rms"]; overlap_a=phase8d.overlap_metrics([x.detach().float().cpu() for x in anchored],regions)["aggregate_rms"]
      dt=sigma_next-sigma; bp_next=blueprint+(blueprint-bp_x0)/sigma*dt; h_next=h+(h-anchored_h)/sigma*dt
      if phase14.tensor_hash(h)!=h_hash or phase14.tensor_hash(blueprint)!=bp_hash or [phase14.tensor_hash(x) for x in working]!=w_hashes: raise RuntimeError("accepted input or W mutated")
      if not all(torch.isfinite(x).all() for x in (bp_next,h_next,ordinary_h,anchored_h)): raise RuntimeError("nonfinite")
      rec={"ordinal":ordinal,"sigma":float(sigma),"sigma_next":float(sigma_next),"accepted_input_H_hash":h_hash,"accepted_input_blueprint_hash":bp_hash,"accepted_output_H_hash":phase14.tensor_hash(h_next),"accepted_output_blueprint_hash":phase14.tensor_hash(bp_next),"blueprint_x0":phase14.summary(bp_x0),"ordinary_x0_H":phase14.summary(ordinary_h),"anchored_x0_H":phase14.summary(anchored_h),"accepted_H":phase14.summary(h_next),"accepted_blueprint":phase14.summary(bp_next),"ordinary_vs_anchored":phase17.tensor_difference(ordinary_h,anchored_h),"anchor":{"correction_rms":rms(anchored_h-ordinary_h),"correction_over_ordinary_local_x0_rms_mean":sum(ratios)/len(ratios),"correction_over_ordinary_local_x0_rms_max":max(ratios),"coarse_discrepancy_before_rms_mean":sum(before)/len(before),"coarse_error_after_max_abs":max(after)},"overlap":{"ordinary_rms":overlap_o,"anchored_rms":overlap_a},"coverage":{"ordinary":[float(cov_o.min()),float(cov_o.max())],"anchored":[float(cov_a.min()),float(cov_a.max())]},"model_calls":{"blueprint":1,"local":55,"total":56},"timing":{"blueprint_cuda_ms":bp_ms,"local_cuda_ms":local_ms,"wall_seconds":time.perf_counter()-wall},"memory":{"peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved())},"integrity":{"accepted_inputs_immutable":True,"W_inputs_immutable":True,"finite":True,"atomic_commit":True}}
      tensors={"blueprint_x0":bp_x0.detach().float().cpu(),"ordinary_x0_H":ordinary_h.detach().float().cpu(),"anchored_x0_H":anchored_h.detach().float().cpu(),"accepted_H":h_next.detach().float().cpu(),"accepted_blueprint":bp_next.detach().float().cpu(),"representative_x0_W":x0_w[27].detach().float().cpu()}
      save_interval(ordinal,key,rec,tensors); h=h_next.detach(); blueprint=bp_next.detach(); records.append(rec); outputs.append({k:tensors[k] for k in ("blueprint_x0","ordinary_x0_H","anchored_x0_H","accepted_H")}); print(f"phase20c interval {ordinal} complete {ordinal+1}/4 wall={rec['timing']['wall_seconds']:.2f}s",flush=True)
    self.result={"configuration":config,"configuration_hash":key,"blueprint_initialization":{"coarse":phase14.summary(bp_coarse),"added_noise":phase14.summary(bp_noise)},"intervals":records,"integrity":{"accepted_intervals":4,"atomic_pair_commits":4,"blueprint_forward_count":4,"local_forward_count":220,"all_anchor_errors_within_tolerance":all(x["anchor"]["coarse_error_after_max_abs"]<=1e-5 for x in records),"all_inputs_immutable":all(x["integrity"]["accepted_inputs_immutable"] and x["integrity"]["W_inputs_immutable"] for x in records),"finite":all(x["integrity"]["finite"] for x in records),"production_changes":False},"final_H":phase14.summary(h),"final_blueprint":phase14.summary(blueprint)}; self.outputs=outputs
    return sampling.inverse_noise_scaling(sigmas[-1],h)

def decode(vae,x,path):
  phase2.save_pixels(vae.decode(x).cpu(),path)
  with Image.open(path) as im: rgb=im.convert("RGB"); return {"path":str(path),"dimensions_wh":list(rgb.size),"sha256_rgb":hashlib.sha256(rgb.tobytes()).hexdigest()}
def sheet(items,path):
  panels=[]
  for label,p in items:
    im=Image.open(p).convert("RGB"); im.thumbnail((2048,512)); panel=Image.new("RGB",(2048,im.height+38),"white"); panel.paste(im,((2048-im.width)//2,38)); ImageDraw.Draw(panel).text((10,10),label,fill="black"); panels.append(panel)
  out=Image.new("RGB",(2048,sum(x.height for x in panels)),"white"); y=0
  for p in panels: out.paste(p,(0,y)); y+=p.height
  out.save(path)
def main():
  OUTPUT.mkdir(parents=True,exist_ok=True); preflight={"estimated_minutes":20,"stop_threshold_minutes":30,"proceed":True,"work":"4 Blueprint + 220 local forwards + 16 requested decodes"}; atomic_json(OUTPUT/"persistent_blueprint_preflight.json",preflight); print(json.dumps({"preflight":preflight}),flush=True)
  model=phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH),model_options={}); clip=phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)],clip_type=phase2.comfy.sd.CLIPType.FLUX2); positive=clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT)); negative=clip.encode_from_tokens_scheduled(clip.tokenize("")); del clip; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
  noise=torch.randn((1,128,*phase14.H_HW),generator=torch.Generator().manual_seed(phase14.SEED)); sigmas=phase2.get_schedule(phase14.STEPS,math.prod(phase14.H_HW)).float().clone(); sigmas[0]=1.0; sampler=Phase20cSampler(); perf.prepare_model_state(model)
  with torch.inference_mode(): phase2.comfy.sample.sample_custom(model,noise,1.0,sampler,sigmas,positive,negative,torch.zeros_like(noise),callback=lambda *args:None,disable_pbar=True,seed=phase14.SEED)
  del model; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache(); vae=phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH),safe_load=True)); decoded={}; items=[]; labels=(("blueprint_x0","BLUEPRINT"),("ordinary_x0_H","ORDINARY"),("anchored_x0_H","ANCHORED"),("accepted_H","ACCEPTED_H"))
  for i,out in enumerate(sampler.outputs):
    for key,label in labels:
      p=OUTPUT/f"PERSISTENT_{i}_{label}.png"; decoded[f"interval_{i}_{key}"]=decode(vae,out[key],p); items.append((f"interval {i} {key}",p))
  sp=OUTPUT/"PERSISTENT_BLUEPRINT_TRAJECTORY_COMPARISON.png"; sheet(items,sp); sampler.result["decoded"]=decoded; sampler.result["comparison_sheet"]=str(sp); sampler.result["preflight"]=preflight; atomic_json(REPORT,sampler.result); print(json.dumps({"report":str(REPORT),"sheet":str(sp)},indent=2),flush=True)
if __name__=="__main__": main()
