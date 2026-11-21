import math

def evaluate(classifier, records, definitions):
    from .schema import pairs
    gold={(r.id,c):y for r in records for c,y in r.labels.items()}
    out={}; totals=[0,0,0]; brier=0; n=0
    for c in definitions:
        tp=fp=fn=tn=0
        for pair in pairs(records,{c:definitions[c]}):
            if (pair.id,c) not in gold: continue
            y=gold[(pair.id,c)]; p=classifier.predict(pair); pred=int(p>=.5)
            tp+=int(pred==1 and y==1); fp+=int(pred==1 and y==0)
            fn+=int(pred==0 and y==1); tn+=int(pred==0 and y==0)
            brier+=(p-y)**2; n+=1
        f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0
        out[c]={"precision":tp/(tp+fp) if tp+fp else 0.,
                "recall":tp/(tp+fn) if tp+fn else None,
                "f1":f1 if tp+fn else None,"support":tp+fn,
                "tp":tp,"fp":fp,"fn":fn,"tn":tn}
        totals=[a+b for a,b in zip(totals,(tp,fp,fn))]
    scores=[v["f1"] for v in out.values() if v["f1"] is not None]
    tp,fp,fn=totals
    return {"per_code":out,"macro_f1_supported":sum(scores)/len(scores) if scores else None,
            "micro_f1":2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.,"brier":brier/n if n else None}
