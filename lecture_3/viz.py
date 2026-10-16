"""Рисунки для практикума: граф зависимостей корпуса."""
import textwrap
import matplotlib.pyplot as plt
import networkx as nx

NODE = 3900

COLORS = {"Раздел": "#E1E6FF", "Термин": "#FFECD6", "Требование": "#E1FAE1"}

def deps_graph(corpus):
    """Ориентированный граф: ребро x → y означает «x зависит от y»."""
    g = nx.DiGraph()
    for _, r in corpus.iterrows():
        g.add_node(r["id"], тип=r["тип"], название=r["название"])
        for d in filter(None, r["зависит_от"].split(";")):
            g.add_edge(r["id"], d.strip())
    return g

def draw_deps(corpus, start=None, show_inferred=False, title="", figsize=(12, 6), short=26,
              node_size=NODE, font_size=6.5, note=True):
    """Рисует зависимости. С start показывает только start и всё, на что он опирается.

    Сплошные стрелки записаны в корпусе. При show_inferred пунктиром добавлены зависимости start,
    которые получаются по транзитивности.
    """
    g = deps_graph(corpus)
    if start:
        g = g.subgraph({start} | nx.descendants(g, start)).copy()
    # слои: чем дальше от «листьев», тем левее
    depth = {}
    for n in reversed(list(nx.topological_sort(g))):
        depth[n] = 1 + max((depth[m] for m in g.successors(n)), default=-1)
    for n, d in depth.items():
        g.nodes[n]["layer"] = -d
    pos = nx.multipartite_layout(g, subset_key="layer", scale=1.0)
    fig, ax = plt.subplots(figsize=figsize)
    def name(n):
        t = g.nodes[n]["название"]
        t = t.split(". ", 1)[-1] if t.startswith("Раздел") else t      # «Раздел 3. Термины» → «Термины»
        t = t.split(" ", 1)[-1] if t[:1].isdigit() else t               # «6.5 Принятие решения» → «Принятие решения»
        return "\n".join(textwrap.wrap(t[:short], 13)[:2])
    labels = {n: f"{n}\n{name(n)}" for n in g}
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#666666", arrowsize=12, width=1.2,
                           node_size=node_size, connectionstyle="arc3,rad=0.06")
    inferred = []
    if start and show_inferred:
        inferred = [(start, n) for n in nx.descendants(g, start) if not g.has_edge(start, n)]
        nx.draw_networkx_edges(g, pos, edgelist=inferred, ax=ax, edge_color="#CD282D", style="dashed",
                               arrowsize=10, width=1.0, node_size=node_size, connectionstyle="arc3,rad=-0.25")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size, node_shape="s", edgecolors="#888888",
                           node_color=[("#FFF8D2" if n == start else COLORS.get(g.nodes[n]["тип"], "#ECECEC")) for n in g])
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=font_size)
    text = f"записано рёбер: {g.number_of_edges()}"
    if start:
        text = (f"{start}: записано зависимостей {g.out_degree(start)}, "
                f"с учётом транзитивности {len(nx.descendants(g, start))}")
    if note or title:
        ax.set_title((title + "\n" if title else "") + (text if note else ""), fontsize=10)
    ax.margins(0.08)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    return len(inferred)
