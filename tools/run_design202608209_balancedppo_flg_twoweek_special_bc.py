#!/usr/bin/env python3
"""Train conservative actor6 special-BC endpoints on two-week FLG data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path: sys.path.insert(0, str(TOOLS))
import torch  # noqa: E402
import run_ppo_bc_repair as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402

EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PARENT = ROOT / "artifacts/design202608198_s16_balancedanchor_ppo1x192_u477/B_gold_league/seed-202608198/checkpoints/update-0477.pt"
PARENT_SHA256 = "a76429ae686fe5ca81582d3c3dc1514c79565db232d723c3bcc1dcec15b74f76"
SPECIAL = ROOT / "data/bc_marnie_flg_current14_trainvalid_through0804_design202608208.zip"
SPECIAL_SHA256 = "e22953532ef1024ac8b926e9ec90977e9fc05ca8bd3d23860e1e7bee1544e766"
GENERAL = ROOT / "data/bc_marnie_top50_current14_latesttop50_trainvalid_through0804_design202608194.zip"
GENERAL_SHA256 = "74ebd086d065f0e2ab95dde2192e463c8ba1eb04ab4e40c1876d484f66475c15"
BC = ROOT / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
BC_SHA256 = "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
TRAINER = TOOLS / "train_ppo.py"
TRAINER_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
SEED = 202608209
BATCH_INDICES = (28, 9, 20, 31, 11, 30, 18, 4)
ENDPOINTS = (2, 4, 8)
SPECIAL_BATCHES = 32
VALID_BATCHES = 16
BATCH_SIZE = 256
LR = 4e-7
ACTOR6 = ("actor_query.weight", "actor_key.weight", "actor_residual.0.weight", "actor_residual.0.bias", "actor_residual.2.weight", "actor_residual.2.bias")
OUTPUT_ROOT = ROOT / "artifacts/design202608209_balancedppo_flg_twoweek_special_bc"
PREFLIGHT_ROOT = ROOT / "artifacts/design202608209_balancedppo_flg_twoweek_special_bc_preflight"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def require(path: Path, expected: str) -> None:
    info=os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or sha(path)!=expected: raise RuntimeError(f"frozen input mismatch: {path}")

def publish(path: Path, raw: bytes) -> None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try: os.write(fd,raw);os.fsync(fd)
    finally: os.close(fd)

@torch.inference_mode()
def cache_loss(model, batches, config, device):
    model.eval();total=0.0;rows=0;parts_total={}
    for cpu in batches:
        batch={k:v.to(device) for k,v in cpu.items()};out=ppo.model_forward(model,batch,device)
        loss,parts=ppo.bc_expert_actor_loss(out,batch,loss_mode=config.bc_replay_loss,order_context_weight=config.bc_replay_order_context_weight,non_context34_fixed_multi_action_order_weight=config.bc_replay_non_context34_fixed_multi_action_order_weight)
        n=int(batch['action_counts'].shape[0]);total+=float(loss)*n;rows+=n
        for k,v in parts.items():parts_total[k]=parts_total.get(k,0.0)+float(v)*n
    result={'loss':total/rows,'rows':rows}|{k:v/rows for k,v in parts_total.items()}
    if not audit.finite_nested(result):raise FloatingPointError('non-finite cache metrics')
    return result

def l2(before,after,names):return math.sqrt(sum(float((after[n].double()-before[n].double()).square().sum()) for n in names))

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-tool-sha256',required=True);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    tool=Path(__file__).resolve()
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=EXPECTED_PYTHON.resolve() or sys.flags.isolated!=1 or sys.flags.dont_write_bytecode!=1:raise RuntimeError('requires repo cwd and my_project_env Python -I -B')
    for path,digest in ((tool,args.expected_tool_sha256),(PARENT,PARENT_SHA256),(SPECIAL,SPECIAL_SHA256),(GENERAL,GENERAL_SHA256),(BC,BC_SHA256),(TRAINER,TRAINER_SHA256)):require(path,digest)
    target=PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target.exists() or target.is_symlink():raise FileExistsError(target)
    if len(BATCH_INDICES)!=8 or len(set(BATCH_INDICES))!=8 or not all(0<=v<SPECIAL_BATCHES for v in BATCH_INDICES):raise RuntimeError('batch contract failed')
    random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda');parent=torch.load(PARENT,map_location='cpu',weights_only=False);bc=torch.load(BC,map_location='cpu',weights_only=False)
    if int(parent.get('update',-1))!=477:raise RuntimeError('PPO parent update mismatch')
    model=ppo.instantiate_model_from_checkpoint(parent,bc,device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    named=dict(model.named_parameters());parameters=[]
    for name in ACTOR6:named[name].requires_grad_(True);parameters.append(named[name])
    before=audit.clone_model_state(model);value_names=[n for n in before if n.startswith('value_head.')];count_names=[n for n in before if n.startswith('count_head.')]
    special_config=ppo.PPOConfig(**parent['config']);special_config.bc_replay_data=str(SPECIAL);special_config.bc_replay_split='train';special_config.bc_replay_batches=SPECIAL_BATCHES;special_config.bc_replay_batch_size=BATCH_SIZE;special_config.bc_replay_workers=2;special_config.bc_replay_steps=1;special_config.bc_replay_lr_scale=1.0;special_config.bc_replay_loss='ordered';special_config.bc_replay_order_context_weight=8.0;special_config.bc_replay_non_context34_fixed_multi_action_order_weight=1.0;special_config.bc_replay_context34_rows_per_batch=1;special_config.seed=SEED
    valid_config=copy.deepcopy(special_config);valid_config.bc_replay_data=str(GENERAL);valid_config.bc_replay_split='valid';valid_config.bc_replay_batches=VALID_BATCHES;valid_config.bc_replay_workers=1;valid_config.bc_replay_context34_rows_per_batch=4;valid_config.seed=SEED+1
    special_batches=ppo.build_bc_replay_batches(special_config,parent['model_config']);valid_batches=ppo.build_bc_replay_batches(valid_config,parent['model_config'])
    special_sha,batch_shas=audit.replay_cache_manifest(special_batches);valid_sha,_=audit.replay_cache_manifest(valid_batches)
    if len(special_batches)!=SPECIAL_BATCHES or len(valid_batches)!=VALID_BATCHES:raise RuntimeError('cache count drift')
    if set(int((b['contexts']==ppo.SKILL_ORDER_CONTEXT).sum()) for b in special_batches)!={1}:raise RuntimeError('special context34 drift')
    special_before=cache_loss(model,special_batches,special_config,device);valid_before=cache_loss(model,valid_batches,valid_config,device)
    common={'schema_version':'ptcg-design202608209-balancedppo-flg-twoweek-special-bc-v1','mode':'audit_only' if args.audit_only else 'special_bc','parent':{'path':str(PARENT.relative_to(ROOT)),'sha256':PARENT_SHA256,'update':477},'special_data':{'path':str(SPECIAL.relative_to(ROOT)),'sha256':SPECIAL_SHA256,'team':'flg','rows':33823,'dates':['2026-07-22','2026-08-04']},'protocol':{'seed':SEED,'steps':8,'endpoints':list(ENDPOINTS),'batch_indices':list(BATCH_INDICES),'learning_rate':LR,'optimizer':'fresh_adamw','trainable_scope':'actor6'},'caches':{'special_train':{'sha256':special_sha,'rows':SPECIAL_BATCHES*BATCH_SIZE,'loss_before':special_before},'general_valid':{'sha256':valid_sha,'rows':VALID_BATCHES*BATCH_SIZE,'loss_before':valid_before,'used_for_optimizer_steps':False}},'scope':{'local_only':True,'package':False,'upload':False,'submission':False}}
    os.mkdir(target,mode=0o700)
    if args.audit_only:
        result=common|{'status':'audit_passed','optimizer_steps':0,'checkpoint_writes':0,'model_state_unchanged':audit.changed_tensor_names(before,audit.clone_model_state(model))==[],'validation_rows_used_for_training':0};publish(target/'preflight_audit.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode());print(json.dumps(result,sort_keys=True));return 0
    optimizer=torch.optim.AdamW(parameters,lr=LR,eps=1e-5,weight_decay=1e-4);per_step=[];endpoints={}
    for step,index in enumerate(BATCH_INDICES,1):
        metrics=ppo.bc_replay_update(model,optimizer,[special_batches[index]],special_config,device,LR)
        if metrics.get('rows')!=BATCH_SIZE or metrics.get('steps')!=1:raise RuntimeError('special step contract failed')
        per_step.append({'step':step,'batch_index':index,'batch_sha256':batch_shas[index],'metrics':metrics})
        if step not in ENDPOINTS:continue
        state=audit.clone_model_state(model);changed=audit.changed_tensor_names(before,state);special_loss=cache_loss(model,special_batches,special_config,device);valid_loss=cache_loss(model,valid_batches,valid_config,device)
        if changed!=sorted(ACTOR6) or any(not torch.equal(before[n],state[n]) for n in value_names+count_names) or special_loss['loss']>=special_before['loss'] or not audit.finite_nested(state):raise RuntimeError(f'S{step} integrity failed')
        payload={k:copy.deepcopy(parent[k]) for k in ('feature_version','bc_feature_version','config','model_config','learner_deck_hash','reward','action_distribution')};payload.update({'model_state_dict':state,'update':477,'evaluation_only':True,'resume_forbidden':True,'optimizer_states_omitted':['optimizer_state_dict','bc_replay_optimizer_state_dict','opponent_quota_state'],'post_ppo_special_bc':{'schema_version':common['schema_version'],'steps':step,'team':'flg','fresh_optimizer':True,'validation_rows_used_for_training':0}})
        buffer=io.BytesIO();torch.save(payload,buffer);raw=buffer.getvalue();checkpoint=OUTPUT_ROOT/f'special-bc-flg-s{step:02d}.pt';publish(checkpoint,raw)
        endpoints[str(step)]={'checkpoint':str(checkpoint.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'actor6_l2':l2(before,state,ACTOR6),'special_loss':special_loss,'general_valid_loss':valid_loss,'general_valid_loss_delta':valid_loss['loss']-valid_before['loss'],'changed_parameter_names':changed,'value_and_count_heads_unchanged':True,'all_finite':True}
    final=audit.clone_model_state(model);result=common|{'status':'special_bc_completed','per_step':per_step,'endpoints':endpoints,'integrity':{'changed_exactly_actor6':audit.changed_tensor_names(before,final)==sorted(ACTOR6),'value_and_count_heads_unchanged':all(torch.equal(before[n],final[n]) for n in value_names+count_names),'parent_unchanged':sha(PARENT)==PARENT_SHA256,'special_data_unchanged':sha(SPECIAL)==SPECIAL_SHA256,'validation_rows_used_for_training':0,'checkpoint_writes':len(endpoints)}}
    integrity=result['integrity']
    if not (integrity['changed_exactly_actor6'] and integrity['value_and_count_heads_unchanged'] and integrity['parent_unchanged'] and integrity['special_data_unchanged'] and integrity['validation_rows_used_for_training']==0 and integrity['checkpoint_writes']==3):raise RuntimeError('terminal integrity failed')
    publish(OUTPUT_ROOT/'special_bc_manifest.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode());print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
