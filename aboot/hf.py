"""Optional local causal-LM + LoRA binary-label policy.

GRPO here is over single binary label actions, NOT free-form reasoning traces.
Probabilities are the normalized logits of token '0' and token '1'. Rollout
sampling and loss use exactly that same policy, temperature 1. No torch is
required unless this backend is chosen. Integration needs validation on GPU.
"""
import json
from pathlib import Path
from .models import group_advantages
from .schema import write_json

class HFClassifier:
    def __init__(self, codes, model_path, lr=1e-5, device="cuda", max_length=1024,
                 lora_rank=8, local_files_only=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        self.torch=torch; self.model_path=model_path; self.device=device; self.max_length=max_length
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=local_files_only)
        ids=[self.tokenizer.encode(s,add_special_tokens=False) for s in ("0","1")]
        if any(len(x)!=1 for x in ids) or ids[0]==ids[1]:
            raise ValueError("This label-only backend needs distinct single-token labels 0 and 1")
        self.label_ids=[x[0] for x in ids]
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=local_files_only,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
        self.model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=lora_rank,
                    lora_alpha=2*lora_rank,lora_dropout=0.,target_modules="all-linear"))
        self.model.to(device); self.model.eval()
        self.optimizer=torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),lr=lr)
        self.options=dict(model_path=model_path,lr=lr,device=device,max_length=max_length,
                          lora_rank=lora_rank,local_files_only=local_files_only)

    def probability(self,pair):
        # Bound context before applying the template so the final answer marker
        # survives truncation. Full-note/context expansion belongs in the reader.
        text_ids=self.tokenizer.encode(pair.text,add_special_tokens=False)
        text=self.tokenizer.decode(text_ids[:max(1,self.max_length-256)])
        prompt=("Does the text support this label? Answer 0 for no or 1 for yes.\n"
                f"Label: {pair.code}\nDefinition: {pair.definition}\nText:\n{text}\nAnswer:\n")
        inputs=self.tokenizer(prompt,return_tensors="pt",truncation=False).to(self.device)
        if inputs.input_ids.shape[1]>self.max_length:
            raise ValueError("Definition/prompt exceeds max_length: increase it")
        logits=self.model(**inputs,use_cache=False).logits[0,-1,self.label_ids].float()
        return self.torch.softmax(logits,dim=-1)[1].clamp(1e-7,1-1e-7)

    def predict(self,pair):
        with self.torch.no_grad(): return float(self.probability(pair))

    def reference(self):
        owner=self
        class Reference:
            def predict(self,pair):
                with owner.model.disable_adapter(): return owner.predict(pair)
        return Reference()

    def sft(self,rows,epochs=1):
        torch=self.torch
        for _ in range(epochs):
            if not rows: return
            self.optimizer.zero_grad()
            for pair,target,weight in rows:
                p=self.probability(pair)
                loss=-weight*(target*torch.log(p)+(1-target)*torch.log1p(-p))/len(rows)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            self.optimizer.step()

    def grpo(self,groups,reference,beta=.01,clip=.2,epochs=1):
        torch=self.torch
        if not groups: return
        self.optimizer.zero_grad()
        for pair,actions,old_p,label in groups:
            pref=reference.predict(pair); p=self.probability(pair)
            a=torch.tensor(actions,device=self.device,dtype=p.dtype)
            adv=torch.tensor(group_advantages(actions,label),device=self.device,dtype=p.dtype)
            prob=torch.where(a.bool(),p,1-p)
            old=torch.where(a.bool(),torch.full_like(a,old_p),torch.full_like(a,1-old_p))
            ratio=prob/old.clamp_min(1e-7)
            surrogate=torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
            kl=p*(torch.log(p)-__import__('math').log(pref))+(1-p)*(torch.log1p(-p)-__import__('math').log1p(-pref))
            ((-surrogate+beta*kl)/len(groups)).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        self.optimizer.step()

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path); self.tokenizer.save_pretrained(path)
        write_json(path/"aboot_backend.json",self.options)

    @classmethod
    def load(cls,path,codes):
        from peft import PeftModel
        options=json.loads((Path(path)/"aboot_backend.json").read_text())
        obj=cls(codes,**options)
        base=obj.model.get_base_model()
        # Remove newly initialized adapter before loading the trained one.
        base=obj.model.unload()
        obj.model=PeftModel.from_pretrained(base,path,is_trainable=False).to(obj.device)
        obj.model.eval()
        return obj
