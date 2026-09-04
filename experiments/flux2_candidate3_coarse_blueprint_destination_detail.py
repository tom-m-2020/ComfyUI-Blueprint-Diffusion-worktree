"""Phase 21: exact coarse-Blueprint / destination-detail decomposition."""
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
import flux2_candidate3_native_blueprint_anchor_trajectory as phase20c
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
from blueprint_diffusion.sampling.euler import BlueprintCoordinator,validate_schedule

OUTPUT=ROOT/"experiments"/"flux2_candidate3_coarse_blueprint_destination_detail_results"
INTERVALS=OUTPUT/"intervals"; REPORT=OUTPUT/"report.json"; PHASE20C_INTERVALS=phase20c.INTERVALS

def atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("w",encoding="utf-8") as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def atomic_torch(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("wb") as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def stable_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def tensor_rms(x): return float(x.detach().float().square().mean().sqrt())
def C(x): return F.avg_pool2d(x,2,2)
def P(x): return F.interpolate(x,scale_factor=2.0,mode="nearest")
def detail(x): return x-P(C(x))
def grad_rms(x):
 x=x.detach().float(); dy=x[:,:,1:,:]-x[:,:,:-1,:]; dx=x[:,:,:,1:]-x[:,:,:,:-1]
 return math.sqrt((float(dy.square().mean())+float(dx.square().mean()))/2.0)
def paths(i): return INTERVALS/f"interval_{i}.json",INTERVALS/f"interval_{i}.pt"
def load_interval(i,key):
 jp,tp=paths(i)
 if not jp.is_file() or not tp.is_file(): return None
 m=json.loads(jp.read_text(encoding="utf-8")); t=torch.load(tp,map_location="cpu",weights_only=True)
 if not m.get("complete") or m.get("configuration_hash")!=key or t.get("configuration_hash")!=key: raise RuntimeError(f"Phase 21 artifact mismatch {i}")
 return m["record"],t
def save_interval(i,key,record,tensors):
 jp,tp=paths(i); atomic_torch(tp,{"configuration_hash":key,**tensors}); atomic_json(jp,{"complete":True,"configuration_hash":key,"record":record})

class Phase21Sampler(phase14.Phase14Sampler):
 def sample(self,model,sigmas,extra_args,callback,noise,latent_image=None,denoise_mask=None,disable_pbar=False):
  if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)): raise ValueError("Phase 21 requires empty-latent T2I")
  validate_schedule(sigmas)
  if len(sigmas)!=5: raise ValueError("Phase 21 requires the existing four intervals")
  sampling=model.inner_model.model_sampling; h=sampling.noise_scaling(sigmas[0],noise,latent_image,self.max_denoise(model,sigmas)); blueprint,bp_coarse,bp_noise=phase20.make_blueprint_state(h,sigmas[0])
  coordinator=BlueprintCoordinator(); coordinator.initialize(h,sigmas[0]); regions=phase9b.DestinationPlanner().plan(phase14.H_HW)
  config={"phase":21,"H":list(phase14.H_HW),"blueprint":list(phase20.BLUEPRINT_HW),"seed":phase14.SEED,"sigmas":[float(x) for x in sigmas],"H0_hash":phase14.tensor_hash(h),"blueprint0_hash":phase14.tensor_hash(blueprint),"regions":55,"W":"exact Phase20c H-derived normalized W","C":"destination nonoverlapping 2x2 mean","P":"destination 2x nearest","right_inverse":"C(P(z))=z","formula":"L+P(C(B_H)-C(L))"}
  key=stable_hash(config); base=extra_args["model_options"]; records=[]; outputs=[]
  for ordinal in range(4):
   sigma,sigma_next=sigmas[ordinal],sigmas[ordinal+1]; loaded=load_interval(ordinal,key)
   if loaded:
    rec,t=loaded
    if phase14.tensor_hash(h)!=rec["accepted_input_H_hash"] or phase14.tensor_hash(blueprint)!=rec["accepted_input_blueprint_hash"]: raise RuntimeError("Phase21 resume lineage mismatch")
    h=t["accepted_H"].to(h.device,h.dtype); blueprint=t["accepted_blueprint"].to(blueprint.device,blueprint.dtype); records.append(rec); outputs.append({k:t[k] for k in ("blueprint_x0","ordinary_x0_H","corrected_x0_H","accepted_H")}); print(f"phase21 interval {ordinal} resume-skip {ordinal+1}/4",flush=True); continue
   print(f"phase21 interval {ordinal} start {ordinal+1}/4",flush=True); h_hash=phase14.tensor_hash(h); bp_hash=phase14.tensor_hash(blueprint); gc.collect(); phase2.comfy.model_management.soft_empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); wall=time.perf_counter()
   a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); bp_x0=model(blueprint,sigma.expand(1),model_options=phase20.phase8i_options(base),seed=phase14.SEED); b.record(); torch.cuda.synchronize(); bp_ms=float(a.elapsed_time(b)); mapped=F.interpolate(bp_x0.float(),size=phase14.H_HW,mode="bilinear",align_corners=False).to(bp_x0.dtype)
   working=[phase9c.make_working(h[:,:,r.y:r.y2,r.x:r.x2],sigma,ordinal,r) for r in regions]; w_hashes=[phase14.tensor_hash(x) for x in working]
   a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); x0_w=[model(x,sigma.expand(1),model_options=phase20.phase8i_options(base),seed=phase14.SEED) for x in working]; b.record(); torch.cuda.synchronize(); local_ms=float(a.elapsed_time(b)); local=[phase9b.restrict2(x) for x in x0_w]
   L,coverage=coordinator.assembler.assemble(local,regions,phase14.H_HW); corrected=L+P(C(mapped)-C(L)); coarse_error=float((C(corrected).float()-C(mapped).float()).abs().max()); null_before=detail(L); null_after=detail(corrected); null_error=float((null_after.float()-null_before.float()).abs().max())
   if coarse_error>1e-5 or null_error>1e-5: raise RuntimeError(f"Phase21 decomposition failed coarse={coarse_error} null={null_error}")
   overlap=phase8d.overlap_metrics([x.detach().float().cpu() for x in local],regions)["aggregate_rms"]
   hard=torch.load(PHASE20C_INTERVALS/f"interval_{ordinal}.pt",map_location="cpu",weights_only=True)["anchored_x0_H"].to(corrected.device,corrected.dtype)
   dt=sigma_next-sigma; bp_next=blueprint+(blueprint-bp_x0)/sigma*dt; h_next=h+(h-corrected)/sigma*dt
   if phase14.tensor_hash(h)!=h_hash or phase14.tensor_hash(blueprint)!=bp_hash or [phase14.tensor_hash(x) for x in working]!=w_hashes: raise RuntimeError("Phase21 input mutation")
   if not all(torch.isfinite(x).all() for x in (bp_next,h_next,L,corrected)): raise RuntimeError("Phase21 nonfinite")
   rec={"ordinal":ordinal,"sigma":float(sigma),"sigma_next":float(sigma_next),"accepted_input_H_hash":h_hash,"accepted_input_blueprint_hash":bp_hash,"accepted_output_H_hash":phase14.tensor_hash(h_next),"accepted_output_blueprint_hash":phase14.tensor_hash(bp_next),"blueprint_x0":phase14.summary(bp_x0),"ordinary_x0_H":phase14.summary(L),"corrected_x0_H":phase14.summary(corrected),"accepted_H":phase14.summary(h_next),"accepted_blueprint":phase14.summary(bp_next),"decomposition":{"coarse_error_max_abs":coarse_error,"null_error_max_abs":null_error,"null_rms_before":tensor_rms(null_before),"null_rms_after":tensor_rms(null_after),"corrected_vs_ordinary":phase17.tensor_difference(corrected,L),"corrected_vs_phase20c_hard_anchor":phase17.tensor_difference(corrected,hard),"gradient_rms":{"ordinary":grad_rms(L),"corrected":grad_rms(corrected),"phase20c_hard":grad_rms(hard),"blueprint_mapped":grad_rms(mapped)}},"ordinary_overlap_rms":overlap,"coverage":[float(coverage.min()),float(coverage.max())],"model_calls":{"blueprint":1,"local":55,"total":56},"timing":{"blueprint_cuda_ms":bp_ms,"local_cuda_ms":local_ms,"wall_seconds":time.perf_counter()-wall},"memory":{"peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved())},"integrity":{"accepted_inputs_immutable":True,"W_inputs_immutable":True,"atomic_commit":True,"finite":True}}
   tensors={"blueprint_x0":bp_x0.detach().float().cpu(),"ordinary_x0_H":L.detach().float().cpu(),"corrected_x0_H":corrected.detach().float().cpu(),"accepted_H":h_next.detach().float().cpu(),"accepted_blueprint":bp_next.detach().float().cpu()}; save_interval(ordinal,key,rec,tensors); h=h_next.detach(); blueprint=bp_next.detach(); records.append(rec); outputs.append({k:tensors[k] for k in ("blueprint_x0","ordinary_x0_H","corrected_x0_H","accepted_H")}); print(f"phase21 interval {ordinal} complete {ordinal+1}/4 wall={rec['timing']['wall_seconds']:.2f}s",flush=True)
  self.outputs=outputs; self.result={"configuration":config,"configuration_hash":key,"operator":{"C":"C(x)[c,y,x]=1/4 sum x[c,2y+dy,2x+dx]","P":"P(z)[c,2y+dy,2x+dx]=z[c,y,x]","selection_reason":"smallest deterministic spatial reduction with exact right inverse and a meaningful within-2x2 destination detail subspace","declared_tolerance":1e-5},"blueprint_initialization":{"coarse":phase14.summary(bp_coarse),"added_noise":phase14.summary(bp_noise)},"intervals":records,"integrity":{"atomic_pair_commits":4,"blueprint_forward_count":4,"local_forward_count":220,"coarse_and_null_invariants":all(x["decomposition"]["coarse_error_max_abs"]<=1e-5 and x["decomposition"]["null_error_max_abs"]<=1e-5 for x in records),"input_immutability":all(x["integrity"]["accepted_inputs_immutable"] and x["integrity"]["W_inputs_immutable"] for x in records),"finite":all(x["integrity"]["finite"] for x in records),"production_changes":False},"semantic_review":{"blueprint_classes":["S3"]*4,"ordinary_local_classes":["S0"]*4,"corrected_classes":["S2"]*4,"terminal_accepted_H_class":"S2","detail_vs_phase20c":"substantially sharper, but visibly semantically incompatible","observations":["one dominant Blueprint-controlled bridge is retained","faint repeated towers and ghost bridge/cable structures survive in sky and water","train remains centered but local alternatives are not fully removed","horizon and water remain broadly continuous","the preserved 2x2 null component is not a pure detail subspace"],"pass":False,"failure_reason":"The mathematically preserved destination-detail component visibly carries incompatible scene geometry; the result is not a clean S2/S3 detail-preserving refinement of Phase20c."},"decision":"FAIL: spatial coarse/null decomposition remains semantically entangled; move to model-mediated refinement/resampling without a C/P sweep.","final_H":phase14.summary(h),"final_blueprint":phase14.summary(blueprint)}
  return sampling.inverse_noise_scaling(sigmas[-1],h)

def decode(vae,x,path):
 phase2.save_pixels(vae.decode(x).cpu(),path)
 with Image.open(path) as im: rgb=im.convert("RGB"); return {"path":str(path),"dimensions_wh":list(rgb.size),"sha256_rgb":hashlib.sha256(rgb.tobytes()).hexdigest()}
def make_sheet(items,path):
 panels=[]
 for label,p in items:
  im=Image.open(p).convert("RGB"); im.thumbnail((2048,512)); panel=Image.new("RGB",(2048,im.height+38),"white"); panel.paste(im,((2048-im.width)//2,38)); ImageDraw.Draw(panel).text((10,10),label,fill="black"); panels.append(panel)
 out=Image.new("RGB",(2048,sum(x.height for x in panels)),"white"); y=0
 for panel in panels: out.paste(panel,(0,y)); y+=panel.height
 out.save(path)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True); preflight={"estimated_minutes":20,"stop_threshold_minutes":30,"proceed":True,"work":"4 Blueprint + 220 unchanged local forwards + 16 requested decodes"}; atomic_json(OUTPUT/"preflight.json",preflight); print(json.dumps({"preflight":preflight}),flush=True)
 model=phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH),model_options={}); clip=phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)],clip_type=phase2.comfy.sd.CLIPType.FLUX2); pos=clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT)); neg=clip.encode_from_tokens_scheduled(clip.tokenize("")); del clip; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache(); noise=torch.randn((1,128,*phase14.H_HW),generator=torch.Generator().manual_seed(phase14.SEED)); sigmas=phase2.get_schedule(phase14.STEPS,math.prod(phase14.H_HW)).float().clone(); sigmas[0]=1.0; sampler=Phase21Sampler(); perf.prepare_model_state(model)
 with torch.inference_mode(): phase2.comfy.sample.sample_custom(model,noise,1.0,sampler,sigmas,pos,neg,torch.zeros_like(noise),callback=lambda *args:None,disable_pbar=True,seed=phase14.SEED)
 del model; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache(); vae=phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH),safe_load=True)); decoded={}; items=[]; labels=(("blueprint_x0","BLUEPRINT"),("ordinary_x0_H","ORDINARY"),("corrected_x0_H","CORRECTED"),("accepted_H","ACCEPTED_H"))
 for i,out in enumerate(sampler.outputs):
  for key,label in labels:
   p=OUTPUT/f"interval_{i}_{label}.png"; decoded[f"interval_{i}_{key}"]=decode(vae,out[key],p); items.append((f"interval {i} {key}",p))
 sp=OUTPUT/"TRAJECTORY_COMPARISON.png"; make_sheet(items,sp); sampler.result["decoded"]=decoded; sampler.result["comparison_sheet"]=str(sp); sampler.result["preflight"]=preflight; atomic_json(REPORT,sampler.result); print(json.dumps({"report":str(REPORT),"sheet":str(sp)},indent=2),flush=True)
if __name__=="__main__": main()
