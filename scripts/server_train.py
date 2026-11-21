#!/usr/bin/env python3
"""Launch one HF/LoRA bootstrap experiment from a clean Git checkout.

Data/weights/credentials stay on the server. No implicit API or GPU selection.
Run from the repository root. Requires installation with pip install -e '.[hf]'.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Server-local ORAB_MIMIC.csv')
    p.add_argument('--model',required=True,help='Server-local causal LM directory')
    p.add_argument('--definitions',required=True,help='Code-to-definition JSON; use task codebook')
    p.add_argument('--output',required=True,help='New experiment directory outside checkout')
    p.add_argument('--mode',choices=['sft','grpo','soft','hybrid'],default='grpo')
    teacher=p.add_mutually_exclusive_group(required=True)
    teacher.add_argument('--oracle-simulation',action='store_true',help='Use TRAIN gold; not GPT')
    teacher.add_argument('--teacher-config',help='Local GPT config JSON, with explicit model and cache path')
    p.add_argument('--gpu',help='Explicit CUDA_VISIBLE_DEVICES; otherwise preserve server environment')
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--budget',type=int,default=180)
    p.add_argument('--round-budget',type=int,default=36)
    p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--selection',choices=['uniform','uniform_bins','central'],default='uniform_bins')
    p.add_argument('--dry-run',action='store_true',help='Print plan without training or API calls')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output).expanduser().resolve()
    for name in ('data','model','definitions'):
        v=Path(getattr(args,name)).expanduser().resolve()
        if not v.exists(): p.error(f'{name} path does not exist: {v}')
        setattr(args,name,str(v))
    if not Path(args.model).is_dir(): p.error('--model must be a local directory')
    if args.rounds<1 or args.budget<0 or args.round_budget<0: p.error('Invalid rounds/budget')
    if output.exists(): p.error('Output path already exists; use a new experiment directory')
    cfg=json.loads((root/'configs'/f'{args.mode}.json').read_text())
    cfg.update(rounds=args.rounds,teacher_budget_total=args.budget,
               teacher_budget_round=args.round_budget,selection=args.selection)
    cfg['classifier']={'backend':'hf','model_path':args.model,'lr':1e-5,'device':'cuda',
                       'max_length':args.max_length,'lora_rank':8,'local_files_only':True}
    if args.teacher_config:
        cfg['teacher']=json.loads(Path(args.teacher_config).read_text())
        if 'cache_path' in cfg['teacher']:
            cfg['teacher']['cache_path']=str(Path(cfg['teacher']['cache_path']).expanduser().resolve())
    prepared=output/'dataset'
    commands=[
        [sys.executable,'-m','aboot.cli','prepare','--reader','odd','--input',args.data,
         '--definitions',args.definitions,'--output',str(prepared)],
        [sys.executable,'-m','aboot.cli','run','--data',str(prepared),'--config',str(output/'config.json'),
         '--output',str(output/'training')]+(['--oracle-simulation'] if args.oracle_simulation else []),
        [sys.executable,'-m','aboot.cli','diagnose','--data',str(prepared),
         '--run',str(output/'training'),'--split','dev']]
    if args.dry_run:
        print(json.dumps({'commands':commands,'classifier':cfg['classifier'],
                          'gpu':args.gpu,'mode':args.mode,'teacher':'oracle' if args.oracle_simulation else 'gpt'},indent=2))
        return
    try:
        import torch
        import transformers
        import peft
    except ImportError:
        p.error("Install dependencies first: python -m pip install -e '.[hf]'")
    # Set before the first CUDA initialization; importing torch alone does not initialize CUDA.
    env=os.environ.copy()
    if args.gpu is not None: env['CUDA_VISIBLE_DEVICES']=args.gpu
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    (output/'source_commit.txt').write_text(result.stdout.strip()+'\n')
    for i,cmd in enumerate(commands):
        print('stage',i+1,flush=True)
        with (output/f'stage_{i+1}.log').open('w') as f:
            subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(f'Completed. Development diagnostics: {output}/training/diagnostics_dev.json')

if __name__=='__main__': main()
