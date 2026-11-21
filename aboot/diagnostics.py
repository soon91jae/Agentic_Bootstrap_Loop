"""Gold-based module diagnostics, separate from acquisition/training.

Each code is a query and all known-labeled contexts in the selected split form
its retrieval corpus. Never pool test contexts into the training index. Unknown
labels are excluded from ranking metrics, never assumed negative. At k, corpus
size is clamped; tie breaks are deterministic by row id, independent of labels.
"""
from pathlib import Path
import json
from .models import TrainableRetriever
from .schema import pairs
from .metrics import evaluate


def ranking_stats(labels, ks):
    support=sum(labels); hits=0; ap=0.
    for rank,y in enumerate(labels,1):
        hits+=y
        if y: ap+=hits/rank
    out={"support":support,"corpus_size":len(labels),
         "average_precision":ap/support if support else None}
    for k in ks:
        limit=min(k,len(labels)); found=sum(labels[:limit])
        out[str(k)]={"effective_k":limit,"hits":found,
                    "precision":found/limit if limit else None,
                    "recall":found/support if support else None}
    return out


def load_retriever(path):
    state=json.loads(Path(path).read_text())
    model=TrainableRetriever({c:"" for c in state['q']},state['dim'],state['lr'])
    model.q=state['q']; model.d=state['d']
    return model


def diagnose(retriever,classifier,records,definitions,ks=(10,25,50,100)):
    ks=sorted(set(ks))
    if not ks or any(k<=0 for k in ks): raise ValueError('k must be positive')
    per_code={}; by_id={r.id:r for r in records}
    for code,definition in definitions.items():
        pool=[p for p in pairs(records,{code:definition}) if code in by_id[p.id].labels]
        ranked=sorted(pool,key=lambda p:(-retriever.score(p),p.id))
        labels=[by_id[p.id].labels[code] for p in ranked]
        stats=ranking_stats(labels,ks); conditional={}; end_to_end={}
        for k in ks:
            selected=ranked[:k]
            predictions={p.id:int(classifier.predict(p)>=.5) for p in selected}
            tp=sum(predictions[p.id]==1 and by_id[p.id].labels[code]==1 for p in selected)
            fp=sum(predictions[p.id]==1 and by_id[p.id].labels[code]==0 for p in selected)
            fn_selected=sum(predictions[p.id]==0 and by_id[p.id].labels[code]==1 for p in selected)
            support=stats['support']; fn_all=support-tp
            def report(fn):
                return {"tp":tp,"fp":fp,"fn":fn,
                    "precision":tp/(tp+fp) if tp+fp else 0.,
                    "recall":tp/(tp+fn) if tp+fn else None,
                    "f1":2*tp/(2*tp+fp+fn) if tp+fn else None}
            conditional[str(k)]=report(fn_selected)
            end_to_end[str(k)]=report(fn_all)
        per_code[code]={"retrieval":stats,"classifier_on_retrieved":conditional,
                        "end_to_end":end_to_end}
    macro={}
    for k in ks:
        vals=[v['retrieval'][str(k)]['recall'] for v in per_code.values()
              if v['retrieval'][str(k)]['recall'] is not None]
        f1s=[v['end_to_end'][str(k)]['f1'] for v in per_code.values()
              if v['end_to_end'][str(k)]['f1'] is not None]
        macro[str(k)]={"retrieval_recall":sum(vals)/len(vals) if vals else None,
                      "end_to_end_f1":sum(f1s)/len(f1s) if f1s else None}
    return {"classifier_full_corpus":evaluate(classifier,records,definitions),
            "per_code":per_code,"macro_at_k":macro,
            "threshold":.5,"tie_break":"row_id","evaluation_rows":len(records)}
