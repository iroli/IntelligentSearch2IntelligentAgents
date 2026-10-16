"""Поиск по словам BM25. Реализация из блокнота лекции 1."""
import collections, math, re

def tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b, self.N = k1, b, len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = collections.Counter(w for d in self.docs for w in set(d))
        self.tf = [collections.Counter(d) for d in self.docs]

    def score(self, query, i):
        s, dl = 0.0, len(self.docs[i])
        for w in tokenize(query):
            f = self.tf[i].get(w, 0)
            if f:
                n = self.df[w]
                idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=5):
        scored = sorted(((self.score(query, i), i) for i in range(self.N)), reverse=True)
        return [(i, s) for s, i in scored[:k] if s > 0]
