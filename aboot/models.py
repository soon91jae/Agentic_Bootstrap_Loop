"""Dependency-free trainable reference models (not pretrained clinical models).

Classifier: class-conditioned logistic policy over hashed word/bigram features.
Retriever: trainable query vectors + shared diagonal document encoder, pairwise
contrastive logistic loss. Both really update parameters; scores are not labels.
"""
import hashlib, math, re
from collections import Counter, defaultdict
from functools import lru_cache
from .schema import sigmoid


@lru_cache(maxsize=20000)
def features(text, dim):
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = words + [a+"_"+b for a,b in zip(words, words[1:])]
    counts = Counter(int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big") % dim
                     for t in tokens)
    norm = math.sqrt(sum(v*v for v in counts.values())) or 1
    return {i:v/norm for i,v in counts.items()}


def dot(w, x): return sum(w[i]*v for i,v in x.items())

class LinearClassifier:
    def __init__(self, codes, dim=512, lr=.2, **kwargs):
        self.dim, self.lr = dim, lr
        self.w = {c:[0.0]*dim for c in codes}; self.b = {c:0.0 for c in codes}

    def predict(self, pair):
        return sigmoid(dot(self.w[pair.code], features(pair.text,self.dim))+self.b[pair.code])

    def sft(self, rows, epochs=1):
        for _ in range(epochs):
            for pair, target, weight in rows:
                p = self.predict(pair); delta = self.lr*weight*(target-p)
                for i,v in features(pair.text,self.dim).items(): self.w[pair.code][i] += delta*v
                self.b[pair.code] += delta

    def grpo(self, groups, reference, beta=.01, clip=.2, epochs=1):
        # All groups collected at one policy snapshot. Batch gradient prevents
        # within-batch updates from silently making later samples off-policy.
        grad = defaultdict(lambda: [defaultdict(float), 0.0])
        for pair, actions, old_p, label in groups:
            p = self.predict(pair); pref = reference.predict(pair)
            advantages = group_advantages(actions, label)
            signal = 0.0
            for a, adv in zip(actions, advantages):
                prob = p if a else 1-p; old = old_p if a else 1-old_p
                ratio = prob/max(old,1e-12)
                if not ((adv>=0 and ratio>1+clip) or (adv<0 and ratio<1-clip)):
                    signal += ratio*adv*(a-p)/len(actions)
            logit_diff = math.log(p/(1-p))-math.log(pref/(1-pref))
            signal -= beta*p*(1-p)*logit_diff
            for i,v in features(pair.text,self.dim).items(): grad[pair.code][0][i] += signal*v
            grad[pair.code][1] += signal
        scale = self.lr/max(1,len(groups))
        for c,(g,b) in grad.items():
            for i,v in g.items(): self.w[c][i] += scale*v
            self.b[c] += scale*b

    def state(self): return {"dim":self.dim,"lr":self.lr,"w":self.w,"b":self.b}


def group_advantages(actions, label):
    rewards = [float(a==label) for a in actions]
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+1e-8) for r in rewards]

class TrainableRetriever:
    def __init__(self, definitions, dim=512, lr=.1):
        self.dim, self.lr = dim, lr
        self.q = {c:[0.0]*dim for c in definitions}; self.d = [1.0]*dim
        for c,text in definitions.items():
            for i,v in features(text,dim).items(): self.q[c][i] = v

    def score(self, pair):
        return sum(self.q[pair.code][i]*self.d[i]*v for i,v in features(pair.text,self.dim).items())

    def train(self, rows, rng, epochs=1):
        grouped = defaultdict(lambda: ([],[]))
        for pair,label in rows: grouped[pair.code][int(label)].append(pair)
        for _ in range(epochs):
            for c,(negative,positive) in grouped.items():
                if not negative or not positive: continue
                for pos in positive:
                    xp=features(pos.text,self.dim); xn=features(rng.choice(negative).text,self.dim)
                    delta={i:xp.get(i,0)-xn.get(i,0) for i in xp.keys()|xn.keys()}
                    margin=sum(self.q[c][i]*self.d[i]*v for i,v in delta.items())
                    step=self.lr*(1-sigmoid(margin))
                    for i,v in delta.items():
                        oldq=self.q[c][i]
                        self.q[c][i] += step*self.d[i]*v
                        self.d[i] += step*oldq*v

    def state(self): return {"dim":self.dim,"lr":self.lr,"q":self.q,"d":self.d}
