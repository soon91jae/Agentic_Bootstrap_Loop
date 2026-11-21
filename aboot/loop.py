import copy, json, random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from .schema import Pair, pairs, key, write_json, write_jsonl
from .models import LinearClassifier, TrainableRetriever
from .selection import sample_actions, select_teacher
from .metrics import evaluate

MODES={"sft","grpo","soft","hybrid"}


def balanced_take(groups, budget, strategy, rng):
    by_code=defaultdict(list)
    for g in groups: by_code[g[0].code].append(g)
    order=list(by_code); rng.shuffle(order)
    selected=[]
    # Redistribute unused quota while maintaining a per-code round-robin.
    queues={c:select_teacher(gs,len(gs),strategy,rng) for c,gs in by_code.items()}
    while len(selected)<budget and any(queues.values()):
        for c in order:
            if len(selected)>=budget: break
            if queues[c]: selected.append(queues[c].pop(0))
    return selected


def candidate_pool(all_pairs, retriever, top_k, explore, rng):
    by_code=defaultdict(list)
    for pair in all_pairs: by_code[pair.code].append(pair)
    result=[]
    for c,ps in by_code.items():
        ranked=sorted(ps,key=lambda p:retriever.score(p),reverse=True)
        chosen=ranked[:top_k]; ids={key(p) for p in chosen}
        remainder=[p for p in ps if key(p) not in ids]
        chosen += rng.sample(remainder,min(explore,len(remainder)))
        result.extend(chosen)
    return result


def run(config, records, definitions, teacher, out):
    mode=config["mode"]
    if mode not in MODES: raise ValueError("Bad mode")
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError("Output directory must be empty (no silent overwrite)")
    out.mkdir(parents=True,exist_ok=True)
    train=[r for r in records if r.split=="train"]
    dev=[r for r in records if r.split=="dev"]
    # Test labels are never read by the training loop or used for model selection.
    pool=pairs(train,definitions); rng=random.Random(config.get("seed",42))
    backend=config.get("classifier",{}).get("backend","linear")
    if backend=="hf":
        from .hf import HFClassifier
        classifier=HFClassifier(definitions,**{k:v for k,v in config["classifier"].items() if k!="backend"})
        reference=classifier.reference()
    elif backend=="linear":
        classifier=LinearClassifier(definitions,**{k:v for k,v in config.get("classifier",{}).items() if k!="backend"})
        reference=copy.deepcopy(classifier)
    else: raise ValueError("Unknown classifier backend")
    retriever=TrainableRetriever(definitions,**config.get("retriever",{}))
    known={}; attempts=set(); stability={}; logs=[]; best=-1
    n=config.get("rollouts",8)
    if n<2: raise ValueError("rollouts must be >=2")
    budget=config.get("teacher_budget_total",200)
    if budget<0: raise ValueError("Negative budget")
    write_json(out/"config.json",config); write_json(out/"definitions.json",definitions)
    write_json(out/"initial_dev.json",evaluate(classifier,dev,definitions))
    for iteration in range(config.get("rounds",5)):
        candidates=candidate_pool(pool,retriever,config.get("top_k_per_code",100),
                                  config.get("explore_per_code",20),rng)
        # The current policy is frozen until all groups and teacher labels collected.
        groups=[]
        for p in candidates:
            prob=classifier.predict(p)
            groups.append((p,sample_actions(prob,n,rng),prob))
        novel=[g for g in groups if key(g[0]) not in known and key(g[0]) not in attempts]
        selected=balanced_take(novel,min(0 if mode=="soft" and iteration>0 else config.get("teacher_budget_round",40),
            max(0,budget-teacher.calls)),config.get("selection","uniform_bins"),rng)
        for pair,actions,prob in selected:
            h=key(pair); attempts.add(h)
            try:
                result=teacher.label(pair)
                if result["status"]=="ok": known[h]=(pair,result["label"],result["provenance"])
            except (ValueError,RuntimeError) as e:
                raise RuntimeError(f"Teacher validation failed for {pair.id}/{pair.code}: {e}") from e
        supervision=[(p,float(y),1.0) for p,y,_ in known.values()]
        grpo_groups=[(p,actions,prob,known[key(p)][1]) for p,actions,prob in groups if key(p) in known]
        pseudo=[]
        threshold=config.get("self_threshold",.95)
        if not .5<threshold<=1: raise ValueError("self_threshold must be >.5 and <=1")
        # Soft targets come from the pre-update policy and are stop-gradient.
        # Asymmetric text dropout on the student prevents the trivially zero
        # gradient from training a model on its identical current prediction.
        for p,actions,prob in (groups if mode in {"soft","hybrid"} else []):
            h=key(p); direction=int(prob>=.5); previous=stability.get(h)
            count=previous[1]+1 if previous and previous[0]==direction else 1
            stability[h]=(direction,count)
            if h not in known and max(prob,1-prob)>=threshold and count>=config.get("stable_rounds",2):
                words=p.text.split()
                kept=[w for w in words if rng.random()>=config.get("text_dropout",.15)]
                noisy=Pair(p.id,p.code,p.definition," ".join(kept) or p.text)
                weight=config.get("self_weight",.25)*(2*abs(prob-.5))
                pseudo.append((noisy,prob,weight))
        if mode=="sft": classifier.sft(supervision,epochs=config.get("sft_epochs",2))
        elif mode=="grpo": classifier.grpo(grpo_groups,reference,beta=config.get("kl_beta",.01))
        elif mode=="soft":
            # Warm-start only: first-round silver labels, then purely soft updates.
            # This arm may spend LESS than the available cap; report actual cost.
            if iteration==0: classifier.sft(supervision,epochs=config.get("sft_epochs",2))
            classifier.sft(pseudo)
        else:
            classifier.grpo(grpo_groups,reference,beta=config.get("kl_beta",.01))
            classifier.sft(pseudo)
        # No self-label feedback into retriever by default: limit error amplification.
        retriever.train([(p,y) for p,y,_ in known.values()],rng,epochs=config.get("retriever_epochs",2))
        metrics=evaluate(classifier,dev,definitions)
        row={"round":iteration+1,"candidate_pairs":len(groups),
             "vote_histogram":dict(Counter(sum(a) for _,a,_ in groups)),
             "new_teacher_selected":len(selected),"teacher_requests":teacher.calls,
             "teacher_tokens":teacher.tokens,"labeled_pairs":len(known),
             "self_training_pairs":len(pseudo),"grpo_groups":len(grpo_groups),"dev":metrics}
        logs.append(row); write_jsonl(out/"rounds.jsonl",logs)
        score=metrics["macro_f1_supported"]
        if score is not None and score>best:
            best=score
            write_json(out/"best_retriever.json",retriever.state())
            classifier.save(out/"best_classifier") if backend=="hf" else write_json(out/"best_classifier.json",classifier.state())
            write_json(out/"best.json",{"round":iteration+1,"dev_macro_f1":best})
        print(json.dumps({k:v for k,v in row.items() if k!="dev"}),flush=True)
    write_jsonl(out/"teacher_labels.jsonl",[{"pair":asdict(p),"label":y,"provenance":source} for p,y,source in known.values()])
    write_json(out/"final_retriever.json",retriever.state())
    classifier.save(out/"final_classifier") if backend=="hf" else write_json(out/"final_classifier.json",classifier.state())
    # No hidden test evaluation here. CLI evaluate is an explicit final operation.
    return logs
