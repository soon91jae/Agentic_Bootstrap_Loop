import math
from collections import defaultdict


def sample_actions(probability, n, rng):
    return [int(rng.random()<probability) for _ in range(n)]


def select_teacher(groups, budget, strategy, rng):
    """groups = (pair, actions, old_p). Endpoints ALWAYS excluded.

    uniform_bins: random draw from occupied vote bins in a shuffled round-robin.
    central: weighted sampling without replacement, weight k*(n-k).
    uniform: uniform over eligible examples, distinct from uniform_bins.
    """
    eligible=[g for g in groups if 0<sum(g[1])<len(g[1])]
    if budget<=0: return []
    if strategy=="uniform":
        return rng.sample(eligible,min(budget,len(eligible)))
    if strategy=="central":
        ranked=[]
        for g in eligible:
            k=sum(g[1]); weight=k*(len(g[1])-k)
            ranked.append((-math.log(max(rng.random(),1e-15))/weight,g))
        ranked.sort(key=lambda x:x[0]); return [g for _,g in ranked[:budget]]
    if strategy=="uniform_bins":
        bins=defaultdict(list)
        for g in eligible: bins[sum(g[1])].append(g)
        for b in bins.values(): rng.shuffle(b)
        out=[]
        while bins and len(out)<budget:
            keys=list(bins); rng.shuffle(keys)
            for k in keys:
                if len(out)>=budget: break
                out.append(bins[k].pop())
                if not bins[k]: del bins[k]
        return out
    raise ValueError(f"Unknown strategy {strategy}")
