"""Полнотекстовый поиск для лекции 5: конвейер индексации, инвертированный индекс,
три ранжирования (TF-IDF с косинусом, BM25, BM25F) и метрики качества.

Код учебный: всё написано явно, без библиотек поиска. Стемминг берёт Snowball из nltk,
если nltk нет, конвейер работает без стемминга.

    from search_bm25 import Pipeline, Index, evaluate, load_qrels
    pipe = Pipeline(stem=True, stop=True)
    idx = Index(docs, pipe, fields={"название": 2.0, "формулировка": 1.0})
    idx.bm25("срок рецензирования", k=5)
"""
import collections
import csv
import math
import re

try:
    from nltk.stem.snowball import SnowballStemmer
    _STEMMER = SnowballStemmer("russian")
except Exception:  # nltk нет: без стемминга
    _STEMMER = None

# Пятьдесят стоп-слов русского языка. Слово «не» намеренно оставлено значимым:
# «не менее двух рецензий» теряет смысл без него.
STOP = set("""и в во на с со к ко по о об от для до из за над под при про без через у а но
или же ли бы то это этот эта эти тот та те так как что чем кто где когда который которая
которые своего свои своих его её их ему ей им их них нём ней""".split())


def tokenize(text):
    """Токены: последовательности букв и цифр в нижнем регистре, ё приводится к е.
    Запись вида 6.5 сохраняется одним токеном."""
    text = text.lower().replace("ё", "е")
    return re.findall(r"\d+(?:\.\d+)+|[а-яa-z0-9]+", text)


class Pipeline:
    """Конвейер индексации: токенизация, стоп-слова, стемминг. Одни и те же шаги
    применяются к документам и к запросам."""

    def __init__(self, stop=True, stem=True, stop_stems=False):
        self.stop = stop
        self.stem = stem and _STEMMER is not None
        self._cache = {}
        # stop_stems=True отбрасывает и основы стоп-слов: «такой» даёт основу «так», как у стоп-слова «так».
        # По умолчанию выключено: стоп-слова убираются до стемминга, как в блокноте лекции.
        self.stop_stems = stop_stems and self.stop and self.stem
        self._stop_stems = {_STEMMER.stem(w) for w in STOP} if self.stop_stems else set()

    def _stem(self, t):
        s = self._cache.get(t)
        if s is None:
            s = self._cache[t] = _STEMMER.stem(t)
        return s

    def __call__(self, text):
        toks = tokenize(text)
        if self.stop:
            toks = [t for t in toks if t not in STOP]
        if self.stem:
            toks = [self._stem(t) for t in toks]
            if self.stop_stems:
                toks = [t for t in toks if t not in self._stop_stems]
        return toks

    def describe(self):
        return "токены" + (" → без стоп-слов" if self.stop else "") + (" → основы Snowball" if self.stem else "") + (" → без основ стоп-слов" if self.stop_stems else "")


class Index:
    """Инвертированный индекс над документами с полями.

    docs: список словарей {поле: текст}, порядок задаёт номера документов.
    fields: {поле: вес} для BM25F. Для обычных TF-IDF и BM25 поля склеиваются в один текст.
    """

    def __init__(self, docs, pipe, fields=None, ids=None):
        self.pipe = pipe
        self.fields = fields or {f: 1.0 for f in docs[0]}
        self.ids = ids or [str(i) for i in range(len(docs))]
        self.N = len(docs)
        # токены по полям и по документу целиком
        self.field_tokens = [{f: pipe(d.get(f, "")) for f in self.fields} for d in docs]
        self.tokens = [sum((ft[f] for f in self.fields), []) for ft in self.field_tokens]
        self.dl = [len(t) for t in self.tokens]
        self.avgdl = sum(self.dl) / self.N
        # инвертированный индекс: слово -> {документ: частота}
        self.postings = collections.defaultdict(dict)
        for i, toks in enumerate(self.tokens):
            for w, f in collections.Counter(toks).items():
                self.postings[w][i] = f
        self.df = {w: len(p) for w, p in self.postings.items()}
        self.vocab = sorted(self.postings)
        # длины полей для BM25F
        self.fdl = {f: [len(ft[f]) for ft in self.field_tokens] for f in self.fields}
        self.avgfdl = {f: sum(self.fdl[f]) / self.N for f in self.fields}

    # ---------------------------------------------------------------- булев поиск
    def docs_with(self, word):
        """Список документов, где есть слово (после конвейера)."""
        w = self.pipe(word)
        return sorted(self.postings.get(w[0], {})) if w else []

    def boolean_and(self, query):
        sets = [set(self.postings.get(w, {})) for w in self.pipe(query)]
        return sorted(set.intersection(*sets)) if sets else []

    # ---------------------------------------------------------------- веса
    def idf_plain(self, w):
        """IDF векторной модели: log(N / df)."""
        return math.log(self.N / self.df[w]) if w in self.df else 0.0

    def idf_bm25(self, w):
        """IDF BM25 с поправками: log((N - df + 0.5) / (df + 0.5) + 1), всегда положителен."""
        n = self.df.get(w, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def tfidf_vector(self, i):
        return {w: f * self.idf_plain(w) for w, f in collections.Counter(self.tokens[i]).items()}

    # ---------------------------------------------------------------- ранжирования
    def _rank(self, scores, k):
        order = sorted(range(self.N), key=lambda i: (-scores[i], i))
        return [(self.ids[i], round(scores[i], 3)) for i in order[:k] if scores[i] > 0]

    def tfidf(self, query, k=5):
        """Косинус между TF-IDF векторами запроса и документа."""
        q = {w: f * self.idf_plain(w) for w, f in collections.Counter(self.pipe(query)).items()}
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores = []
        for i in range(self.N):
            d = self.tfidf_vector(i)
            dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
            scores.append(sum(q[w] * d.get(w, 0.0) for w in q) / (qn * dn))
        return self._rank(scores, k)

    def bm25_score(self, query, i, k1=1.5, b=0.75, explain=False):
        """Оценка BM25 документа i. При explain=True возвращает вклад каждого слова."""
        parts = {}
        tf = collections.Counter(self.tokens[i])
        norm = 1 - b + b * self.dl[i] / self.avgdl
        for w in self.pipe(query):
            f = tf.get(w, 0)
            if f:
                parts[w] = self.idf_bm25(w) * f * (k1 + 1) / (f + k1 * norm)
        return parts if explain else sum(parts.values())

    def bm25(self, query, k=5, k1=1.5, b=0.75):
        """Оценки накапливаются по спискам вхождений слов запроса: документы без слов запроса не трогаются."""
        scores = collections.defaultdict(float)
        for w in self.pipe(query):
            idf = self.idf_bm25(w)
            for i, f in self.postings.get(w, {}).items():
                norm = 1 - b + b * self.dl[i] / self.avgdl
                scores[i] += idf * f * (k1 + 1) / (f + k1 * norm)
        order = sorted(scores, key=lambda i: (-scores[i], i))
        return [(self.ids[i], round(scores[i], 3)) for i in order[:k] if scores[i] > 0]

    def bm25f_score(self, query, i, k1=1.5, b=0.75):
        """BM25F: частоты по полям складываются с весами, длина каждого поля нормируется отдельно."""
        s = 0.0
        for w in self.pipe(query):
            f_tilde = 0.0
            for f, wgt in self.fields.items():
                tf = self.field_tokens[i][f].count(w)
                if tf:
                    norm = 1 - b + b * self.fdl[f][i] / (self.avgfdl[f] or 1)
                    f_tilde += wgt * tf / norm
            if f_tilde:
                s += self.idf_bm25(w) * f_tilde * (k1 + 1) / (f_tilde + k1)
        return s

    def bm25f(self, query, k=5, k1=1.5, b=0.75):
        cand = set()
        for w in self.pipe(query):
            cand.update(self.postings.get(w, {}))
        scores = {i: self.bm25f_score(query, i, k1, b) for i in cand}
        order = sorted(scores, key=lambda i: (-scores[i], i))
        return [(self.ids[i], round(scores[i], 3)) for i in order[:k] if scores[i] > 0]


# -------------------------------------------------------------------- метрики
def load_qrels(path):
    """Читает выборку: id, запрос, релевантные в виде 'R65:2;R61:1', источник.
    Возвращает список (id, запрос, {документ: градация})."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = {}
            for item in row["релевантные"].split(";"):
                d, g = item.split(":")
                rel[d.strip()] = int(g)
            out.append((row["id"], row["запрос"], rel))
    return out


def precision_at_k(ranked, rel, k):
    top = ranked[:k]
    return sum(1 for d in top if rel.get(d, 0) > 0) / k


def recall_at_k(ranked, rel, k):
    total = sum(1 for g in rel.values() if g > 0)
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / total if total else 0.0


def reciprocal_rank(ranked, rel):
    for pos, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            return 1 / pos
    return 0.0


def average_precision(ranked, rel):
    hits, s = 0, 0.0
    for pos, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / pos
    total = sum(1 for g in rel.values() if g > 0)
    return s / total if total else 0.0


def dcg(gains, k):
    return sum(g / math.log2(pos + 1) for pos, g in enumerate(gains[:k], 1))


def ndcg_at_k(ranked, rel, k):
    gains = [rel.get(d, 0) for d in ranked]
    ideal = sorted(rel.values(), reverse=True)
    idcg = dcg(ideal, k)
    return dcg(gains, k) / idcg if idcg else 0.0


def evaluate(search, qrels, k_prec=5, k_ndcg=10):
    """search: функция запрос -> список (id, оценка). Возвращает средние метрики по выборке."""
    m = collections.defaultdict(float)
    for _, q, rel in qrels:
        ranked = [d for d, _ in search(q)]
        m["P@%d" % k_prec] += precision_at_k(ranked, rel, k_prec)
        m["R@%d" % k_prec] += recall_at_k(ranked, rel, k_prec)
        m["MRR"] += reciprocal_rank(ranked, rel)
        m["MAP"] += average_precision(ranked, rel)
        m["NDCG@%d" % k_ndcg] += ndcg_at_k(ranked, rel, k_ndcg)
    return {name: round(v / len(qrels), 3) for name, v in m.items()}

# -------------------------------------------------------------------- совместимость с лекцией 1
def _tokenize_l1(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


class BM25:
    """Класс из блокнота лекции 1, без конвейера: его используют agent_l1.py и agent_my.py.
    Оставлен без изменений, чтобы протоколы агентов лекций 1–3 воспроизводились теми же числами.
    Для лекции 5 используется Index с конвейером."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [_tokenize_l1(d) for d in docs]
        self.k1, self.b, self.N = k1, b, len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = collections.Counter(w for d in self.docs for w in set(d))
        self.tf = [collections.Counter(d) for d in self.docs]

    def score(self, query, i):
        s, dl = 0.0, len(self.docs[i])
        for w in _tokenize_l1(query):
            f = self.tf[i].get(w, 0)
            if f:
                n = self.df[w]
                idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=5):
        scored = sorted(((self.score(query, i), i) for i in range(self.N)), reverse=True)
        return [(i, s) for s, i in scored[:k] if s > 0]
