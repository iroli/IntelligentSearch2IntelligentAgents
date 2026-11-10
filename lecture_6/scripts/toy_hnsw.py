"""Учебная реализация HNSW на чистом Python (лекция 6).

Держит граф по слоям, вставляет точки и ищет ближайших соседей так, как описано в статье
Малкова и Яшунина (препринт 2016, журнальная версия 2020): уровень точки случаен, верхние слои проходятся жадно, на нижнем
слое идёт поиск с очередью размера ef. Записывает путь поиска, чтобы его можно было нарисовать.
Это учебный код: он медленный и нужен только для понимания. По умолчанию соседи выбираются простым правилом (M ближайших), simple_select=False включает эвристику разнообразия из статьи. В работе используется faiss.IndexHNSWFlat.
"""
import heapq, math, random
import numpy as np


class ToyHNSW:
    def __init__(self, M=4, ef_construction=16, seed=0, simple_select=True):
        self.M, self.M0, self.efc = M, 2 * M, ef_construction
        self.mL = 1 / math.log(M)
        self.rng = random.Random(seed)
        self.X = []; self.level = []; self.links = []   # links[i][l] = список соседей точки i на слое l
        self.entry = None; self.top = -1
        self.simple = simple_select
        self.dist_calls = 0

    def dist(self, a, q):
        self.dist_calls += 1
        d = self.X[a] - q
        return float(d @ d)

    def random_level(self):
        return int(-math.log(1 - self.rng.random()) * self.mL)

    def search_layer(self, q, entries, ef, l, trace=None):
        """Поиск на одном слое: кандидаты по возрастанию расстояния, результат держит ef лучших."""
        visited = set(entries)
        cand = [(self.dist(e, q), e) for e in entries]; heapq.heapify(cand)
        res = [(-d, e) for d, e in cand]; heapq.heapify(res)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -res[0][0]:
                break
            if trace is not None: trace.append((l, c))
            for n in self.links[c][l]:
                if n in visited: continue
                visited.add(n); dn = self.dist(n, q)
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, n)); heapq.heappush(res, (-dn, n))
                    if len(res) > ef: heapq.heappop(res)
        return sorted((-d, e) for d, e in res)

    def select(self, q, cands, m):
        if self.simple:
            return [e for _, e in sorted(cands)[:m]]
        # эвристика разнообразия: сосед берётся, если он ближе к q, чем к уже выбранным
        out = []
        for d, e in sorted(cands):
            if all(self.dist(e, self.X[o]) > d for o in out):
                out.append(e)
            if len(out) == m: break
        return out

    def add(self, x):
        i = len(self.X); self.X.append(np.asarray(x, dtype=float))
        L = self.random_level(); self.level.append(L); self.links.append([[] for _ in range(L + 1)])
        if self.entry is None:
            self.entry, self.top = i, L; return i
        ep = [self.entry]
        for l in range(self.top, L, -1):                      # жадный спуск по верхним слоям
            ep = [self.search_layer(self.X[i], ep, 1, l)[0][1]]
        for l in range(min(L, self.top), -1, -1):
            W = self.search_layer(self.X[i], ep, self.efc, l)
            nb = self.select(self.X[i], W, self.M)
            self.links[i][l] = list(nb)
            cap = self.M0 if l == 0 else self.M
            for n in nb:                                       # связи в обе стороны с обрезкой
                self.links[n][l].append(i)
                if len(self.links[n][l]) > cap:
                    c = [(self.dist(m, self.X[n]), m) for m in self.links[n][l]]
                    self.links[n][l] = self.select(self.X[n], c, cap)
            ep = [e for _, e in W]
        if L > self.top:
            self.entry, self.top = i, L
        return i

    def search(self, q, k=1, ef=8, trace=None):
        q = np.asarray(q, dtype=float); ep = [self.entry]
        for l in range(self.top, 0, -1):
            ep = [self.search_layer(q, ep, 1, l, trace)[0][1]]
        W = self.search_layer(q, ep, max(ef, k), 0, trace)
        return [e for _, e in W[:k]]


def recall_vs_ef(X, Q, k=5, efs=(1, 2, 4, 8, 16, 32), M=4, efc=16, seed=0, simple_select=True):
    """Полнота@k учебного HNSW против точного перебора для разных ef. Возвращает список (ef, полнота, расстояний на запрос)."""
    h = ToyHNSW(M=M, ef_construction=efc, seed=seed, simple_select=simple_select)
    for x in X: h.add(x)
    D = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1); gt = np.argsort(D, 1)[:, :k]
    out = []
    for ef in efs:
        h.dist_calls = 0
        got = [h.search(q, k, ef) for q in Q]
        r = np.mean([len(set(g) & set(t)) / k for g, t in zip(got, gt)])
        out.append((ef, float(r), h.dist_calls / len(Q)))
    return h, out
