#!/usr/bin/env python3

# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs

from __future__ import annotations

import argparse
import itertools
import random
import sys
import types
from pathlib import Path

import typing as _typing
import typing_extensions as _typing_extensions

if not hasattr(_typing_extensions, "TypeIs"):
    if hasattr(_typing, "TypeIs"):
        _typing_extensions.TypeIs = _typing.TypeIs  # type: ignore[attr-defined]
    else:
        _typing_extensions.TypeIs = _typing.TypeGuard  # type: ignore[attr-defined]

import numpy as np
import pandas as pd

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined, assignment]

REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "graspe" / "src" / "graspe"
if not REPO_ROOT.is_dir():
    raise SystemExit(
        f"Nedostaje klon graspe: {REPO_ROOT}\n"
        "git clone https://github.com/graphsinspace/graspe.git third_party/graspe"
    )

if "dgl" not in sys.modules:
    _dgl = types.ModuleType("dgl")

    class _DGLGraph:
        pass

    _dgl.DGLGraph = _DGLGraph
    _dgl.from_networkx = lambda *a, **k: None
    sys.modules["dgl"] = _dgl

sys.path.insert(0, str(REPO_ROOT))

import networkx as nx  # noqa: E402
from common.graph import Graph as GraspeGraph  # noqa: E402
from embeddings.embedding_randw import (  # noqa: E402
    HubWalkDistribution,
    HubWalkUniform,
    SCWalk,
    UnbiasedWalk,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CSV = _DATA / "loto7hh_4586_k24.csv"
DEFAULT_COMBOS = _DATA / "kombinacijeH_39C7.csv"
SEED = 39

# Podešavanje (mali graf 39 čvorova)
TUNED_DECAY = 0.999
TUNED_DIM = 32
TUNED_PATH_NUMBER = 55  # broj šetnji *po polaznom čvoru* u deepwalk NX (ukupno ~ path_number * 39)
TUNED_PATH_LENGTH = 14
TUNED_NUM_WALKS_UNBIASED = 56
TUNED_WALK_LENGTH_UNBIASED = 14
TUNED_TOP_NODES = 15
TUNED_W2V_EPOCHS = 18
TUNED_W_GRAPH = 0.42
TUNED_W_EMB = 0.58
TUNED_EXTRA_RANDW_NUM_WALKS = 52
TUNED_EXTRA_RANDW_WALK_LENGTH = 14
TUNED_EXTRA_RANDW_P = 0.85


def load_draws(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = [f"Num{i}" for i in range(1, 8)]
    if all(c in df.columns for c in cols):
        use = cols
    else:
        use = list(df.columns[:7])
    draws = []
    for _, row in df.iterrows():
        draws.append(sorted(int(row[c]) for c in use))
    return draws


def dynamic_pair_weights(draws: list[list[int]], decay: float) -> dict[tuple[int, int], float]:
    T = len(draws)
    acc: dict[tuple[int, int], float] = {}
    for t, nums in enumerate(draws):
        w = float(decay) ** (T - 1 - t)
        for u, v in itertools.combinations(nums, 2):
            a, b = (u, v) if u < v else (v, u)
            acc[(a, b)] = acc.get((a, b), 0.0) + w
    return acc


def build_graspe_graph(pair_w: dict[tuple[int, int], float]) -> GraspeGraph:
    G = GraspeGraph()
    for i in range(1, 40):
        G.add_node(i)
    for (u, v), w in pair_w.items():
        if w <= 0:
            continue
        G.add_edge(u, v, weight=w)
        G.add_edge(v, u, weight=w)
    return G


def marginal_node_weights(draws: list[list[int]], decay: float) -> dict[int, float]:
    T = len(draws)
    acc = {n: 0.0 for n in range(1, 40)}
    for t, nums in enumerate(draws):
        w = float(decay) ** (T - 1 - t)
        for n in nums:
            acc[n] += w
    return acc


def build_graspe_graph_labeled(
    pair_w: dict[tuple[int, int], float], node_label: dict[int, int]
) -> GraspeGraph:
    G = GraspeGraph()
    for i in range(1, 40):
        G.add_node(i, label=int(node_label[i]))
    for (u, v), w in pair_w.items():
        if w <= 0:
            continue
        G.add_edge(u, v, weight=w)
        G.add_edge(v, u, weight=w)
    return G


def synthetic_labels_for_label_walks(
    mode: str, draws: list[list[int]], decay: float
) -> dict[int, int]:
    m = mode.lower().strip()
    if m == "mod7":
        return {n: (n - 1) % 7 for n in range(1, 40)}
    if m == "mod5":
        return {n: (n - 1) % 5 for n in range(1, 40)}
    if m == "decile":
        marg = marginal_node_weights(draws, decay)
        order = sorted(range(1, 40), key=lambda x: (marg[x], x))
        labels: dict[int, int] = {}
        for rank, n in enumerate(order):
            labels[n] = min(9, int(10 * rank / max(1, len(order) - 1)))
        return labels
    raise ValueError(f"nepoznat --label-mode: {mode!r}")


def run_label_aware_randw(
    G: GraspeGraph,
    walk_kind: str,
    dim: int,
    num_walks: int,
    walk_length: int,
    p: float,
    seed: int,
) -> dict[int, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    k = walk_kind.lower().strip()
    if k == "sc":
        emb = SCWalk(
            G, d=dim, num_walks=num_walks, walk_length=walk_length, p=p, workers=1, seed=seed
        )
    elif k == "hub_u":
        emb = HubWalkUniform(
            G, d=dim, num_walks=num_walks, walk_length=walk_length, p=p, workers=1, seed=seed
        )
    elif k == "hub_d":
        emb = HubWalkDistribution(
            G, d=dim, num_walks=num_walks, walk_length=walk_length, p=p, workers=1, seed=seed
        )
    else:
        raise ValueError(walk_kind)
    emb.embed()
    return {n: np.asarray(emb[n], dtype=np.float64).copy() for n in range(1, 40)}


def _to_undirected_nx(G: GraspeGraph) -> nx.Graph:
    dg = G.to_networkx()
    return dg.to_undirected()


def deepwalk_corpus_nx(
    G_undir: nx.Graph,
    path_number: int,
    path_length: int,
    rng: random.Random,
) -> list[list[str]]:
    """DeepWalk-stil korpus: path_number šetnji iz svakog čvora."""
    nodes = list(G_undir.nodes())
    walks: list[list[str]] = []
    for _ in range(path_number):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(path_length - 1):
                nbrs = list(G_undir.neighbors(walk[-1]))
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append([str(x) for x in walk])
    return walks


def _word2vec_from_walks(
    walks: list[list[str]], dim: int, seed: int
) -> dict[int, np.ndarray]:
    import gensim
    from gensim.models import Word2Vec

    major = int(gensim.__version__.split(".")[0])
    workers = 1
    if major >= 4:
        model = Word2Vec(
            walks,
            vector_size=dim,
            window=5,
            min_count=0,
            sg=1,
            workers=workers,
            seed=seed,
            epochs=TUNED_W2V_EPOCHS,
        )
    else:
        model = Word2Vec(
            walks,
            size=dim,
            window=5,
            min_count=0,
            sg=1,
            workers=workers,
            seed=seed,
        )
    wv = model.wv
    out: dict[int, np.ndarray] = {}
    for n in range(1, 40):
        s = str(n)
        out[n] = np.asarray(wv[s], dtype=np.float64) if s in wv else np.zeros(dim, dtype=np.float64)
    return out


def run_deepwalk_nx(
    G: GraspeGraph, dim: int, path_number: int, path_length: int, seed: int
) -> dict[int, np.ndarray]:
    rng = random.Random(seed)
    und = _to_undirected_nx(G)
    walks = deepwalk_corpus_nx(und, path_number, path_length, rng)
    return _word2vec_from_walks(walks, dim, seed)


def _patch_randw_word2vec_epochs() -> None:
    """Dodaj epochs u RWEmbBase.embed za gensim 4."""
    import gensim
    from gensim.models import Word2Vec

    import embeddings.embedding_randw as rw

    _orig = rw.RWEmbBase.embed

    def embed_patched(self) -> None:
        rw.Embedding.embed(self)
        walks = self.simulate_walks()
        walks = [list(map(str, w)) for w in walks]
        major = int(gensim.__version__.split(".")[0])
        workers = max(1, int(self._workers))
        if major >= 4:
            model = Word2Vec(
                sentences=walks,
                vector_size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
                epochs=TUNED_W2V_EPOCHS,
            )
        else:
            model = Word2Vec(
                sentences=walks,
                size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
            )
        self._embedding = {}
        for node in self._g.nodes():
            self._embedding[node[0]] = np.asarray(
                model.wv[str(node[0])], dtype=np.float64
            )

    rw.RWEmbBase.embed = embed_patched


_patch_randw_word2vec_epochs()


def run_unbiased_walk(
    G: GraspeGraph, dim: int, num_walks: int, walk_length: int, seed: int
) -> dict[int, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    emb = UnbiasedWalk(
        G,
        d=dim,
        num_walks=num_walks,
        walk_length=walk_length,
        workers=1,
        seed=seed,
    )
    emb.embed()
    return {n: emb[n].copy() for n in range(1, 40)}


def l2_normalize_rows(vecs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out = {}
    for k, v in vecs.items():
        n = np.linalg.norm(v)
        out[k] = (v / n).astype(np.float64) if n > 1e-12 else v.copy()
    return out


def ensemble_vectors(
    a: dict[int, np.ndarray], b: dict[int, np.ndarray], dim: int
) -> dict[int, np.ndarray]:
    a_n = l2_normalize_rows(a)
    b_n = l2_normalize_rows(b)
    return {n: (a_n[n] + b_n[n]) / 2.0 for n in range(1, 40)}


def ensemble_three_vectors(
    a: dict[int, np.ndarray],
    b: dict[int, np.ndarray],
    c: dict[int, np.ndarray],
    dim: int,
) -> dict[int, np.ndarray]:
    return ensemble_vectors(ensemble_vectors(a, b, dim), c, dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pair_scores(
    pair_w: dict[tuple[int, int], float],
    vectors: dict[int, np.ndarray],
    w_graph: float = TUNED_W_GRAPH,
    w_emb: float = TUNED_W_EMB,
) -> dict[tuple[int, int], float]:
    if not pair_w:
        return {}
    mx = max(pair_w.values()) or 1.0
    scores = {}
    for (u, v), w in pair_w.items():
        gnorm = w / mx
        c = cosine(vectors[u], vectors[v])
        scores[(u, v)] = w_graph * gnorm + w_emb * c
    return scores


def best_combo_from_scores(
    scores: dict[tuple[int, int], float],
    top_nodes: int = TUNED_TOP_NODES,
) -> tuple[int, ...]:
    strength = {n: 0.0 for n in range(1, 40)}
    for (u, v), s in scores.items():
        strength[u] += s
        strength[v] += s
    ranked = sorted(range(1, 40), key=lambda x: (-strength[x], x))[:top_nodes]
    best: tuple[int, ...] | None = None
    best_val = -1e18
    for combo in itertools.combinations(sorted(ranked), 7):
        sv = 0.0
        for u, v in itertools.combinations(combo, 2):
            a, b = (u, v) if u < v else (v, u)
            sv += scores.get((a, b), 0.0)
        if sv > best_val or (sv == best_val and best is not None and combo < best):
            best_val = sv
            best = combo
    assert best is not None
    return best


def main():
    ap = argparse.ArgumentParser(
        description="GRASP loto: DeepWalk NX + UnbiasedWalk — graspe2_loto_walks.py"
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--combos", type=Path, default=DEFAULT_COMBOS)
    ap.add_argument("--decay", type=float, default=TUNED_DECAY)
    ap.add_argument("--dim", type=int, default=TUNED_DIM)
    ap.add_argument(
        "--method",
        type=str,
        default="ensemble",
        choices=("deepwalk", "unbiased", "ensemble", "sc", "hub_u", "hub_d", "mega"),
        help="mega = L2-prosek DeepWalk + UnbiasedWalk + SCWalk (labele iz --label-mode)",
    )
    ap.add_argument(
        "--label-mode",
        type=str,
        default="mod7",
        choices=("mod7", "mod5", "decile"),
        help="Samo za sc | hub_u | hub_d | mega: sintetičke labele za graspe šetnje",
    )
    ap.add_argument(
        "--path-number",
        type=int,
        default=TUNED_PATH_NUMBER,
        help="DeepWalk: koliko puta se iz svakog čvora startuje (puta 39 ukupnih startova po rundama)",
    )
    ap.add_argument("--path-length", type=int, default=TUNED_PATH_LENGTH, help="DeepWalk dužina šetnje")
    ap.add_argument("--num-walks", type=int, default=TUNED_NUM_WALKS_UNBIASED, help="UnbiasedWalk po čvoru")
    ap.add_argument("--walk-length", type=int, default=TUNED_WALK_LENGTH_UNBIASED)
    ap.add_argument("--top-nodes", type=int, default=TUNED_TOP_NODES)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    draws = load_draws(args.csv)
    pair_w = dynamic_pair_weights(draws, args.decay)
    G = build_graspe_graph(pair_w)

    print(f"CSV izvučenih: {args.csv.resolve()}")
    print(f"CSV svih komb.: {args.combos.resolve()}  (postoji: {args.combos.is_file()})")
    print(f"Izvlačenja: {len(draws)} | parova: {len(pair_w)} | decay={args.decay}")
    print(f"graspe: {REPO_ROOT}")
    print(
        f"Metoda: {args.method}"
        + (
            f" | label-mode={args.label_mode}"
            if args.method in ("sc", "hub_u", "hub_d", "mega")
            else ""
        )
    )

    if args.method == "deepwalk":
        vectors = run_deepwalk_nx(
            G, args.dim, args.path_number, args.path_length, SEED
        )
        label = "DeepWalk (NX korpus + W2V)"
    elif args.method == "unbiased":
        vectors = run_unbiased_walk(
            G, args.dim, args.num_walks, args.walk_length, SEED
        )
        label = "UnbiasedWalk (graspe embedding_randw)"
    elif args.method == "ensemble":
        v_dw = run_deepwalk_nx(G, args.dim, args.path_number, args.path_length, SEED)
        v_ub = run_unbiased_walk(G, args.dim, args.num_walks, args.walk_length, SEED)
        vectors = ensemble_vectors(v_dw, v_ub, args.dim)
        label = "Ensemble (L2-norm prosek DeepWalk + UnbiasedWalk)"
    elif args.method in ("sc", "hub_u", "hub_d"):
        labs = synthetic_labels_for_label_walks(args.label_mode, draws, args.decay)
        g_l = build_graspe_graph_labeled(pair_w, labs)
        vectors = run_label_aware_randw(
            g_l,
            args.method,
            args.dim,
            TUNED_EXTRA_RANDW_NUM_WALKS,
            TUNED_EXTRA_RANDW_WALK_LENGTH,
            TUNED_EXTRA_RANDW_P,
            SEED,
        )
        label = f"{args.method.upper()} (graspe embedding_randw, labels={args.label_mode})"
    else:
        labs = synthetic_labels_for_label_walks(args.label_mode, draws, args.decay)
        g_l = build_graspe_graph_labeled(pair_w, labs)
        v_dw = run_deepwalk_nx(G, args.dim, args.path_number, args.path_length, SEED)
        v_ub = run_unbiased_walk(G, args.dim, args.num_walks, args.walk_length, SEED)
        v_sc = run_label_aware_randw(
            g_l,
            "sc",
            args.dim,
            TUNED_EXTRA_RANDW_NUM_WALKS,
            TUNED_EXTRA_RANDW_WALK_LENGTH,
            TUNED_EXTRA_RANDW_P,
            SEED + 31,
        )
        vectors = ensemble_three_vectors(v_dw, v_ub, v_sc, args.dim)
        label = f"Mega (DW+UB+SC, labels={args.label_mode})"

    scores = pair_scores(pair_w, vectors)
    combo = best_combo_from_scores(scores, top_nodes=args.top_nodes)

    print()
    print(f"Predikcija ({label}):")
    print(list(combo))
    print()


if __name__ == "__main__":
    main()




## Pokretanje

"""
python3 graspe2_loto_walks.py

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: ensemble

Predikcija (Ensemble (L2-norm prosek DeepWalk + UnbiasedWalk)):
[8, 10, 11, 23, 25, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method ensemble

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: ensemble

Predikcija (Ensemble (L2-norm prosek DeepWalk + UnbiasedWalk)):
[8, 10, 11, 23, 25, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method deepwalk

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: ensemble

Predikcija (Ensemble (L2-norm prosek DeepWalk + UnbiasedWalk)):
[8, 10, 11, 23, 25, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method unbiased

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: unbiased

Predikcija (UnbiasedWalk (graspe embedding_randw)):
[8, 10, 11, 23, 25, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method sc

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: sc | label-mode=mod7

Predikcija (SC (graspe embedding_randw, labels=mod7)):
[8, 11, 22, 25, 29, 32, 39]
"""




"""
python3 graspe2_loto_walks.py --method sc --label-mode mod7

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: sc | label-mode=mod7

Predikcija (SC (graspe embedding_randw, labels=mod7)):
[8, 11, 22, 25, 29, 32, 39]
"""




"""
python3 graspe2_loto_walks.py --method hub_u

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: hub_u | label-mode=mod7

Predikcija (HUB_U (graspe embedding_randw, labels=mod7)):
[8, 10, 23, 25, 32, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method hub_u --label-mode mod7

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: hub_u | label-mode=mod7

Predikcija (HUB_U (graspe embedding_randw, labels=mod7)):
[8, 10, 23, 25, 32, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method hub_d

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: hub_d | label-mode=mod7

Predikcija (HUB_D (graspe embedding_randw, labels=mod7)):
[8, 10, 11, 22, 23, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method hub_d --label-mode mod7

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: hub_d | label-mode=mod7

Predikcija (HUB_D (graspe embedding_randw, labels=mod7)):
[8, 10, 11, 22, 23, 34, 39]
"""




"""
python3 graspe2_loto_walks.py --method mega

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: mega | label-mode=mod7

Predikcija (Mega (DW+UB+SC, labels=mod7)):
[5, 16, 19, 23, 26, 33, 37]
"""




"""
python3 graspe2_loto_walks.py --method mega --label-mode mod7

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: mega | label-mode=mod7

Predikcija (Mega (DW+UB+SC, labels=mod7)):
[5, 16, 19, 23, 26, 33, 37]
"""




"""
python3 graspe2_loto_walks.py --method mega --label-mode decile

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Metoda: mega | label-mode=decile

Predikcija (Mega (DW+UB+SC, labels=decile)):
[8, 10, 23, 25, 29, 34, 37]
"""



########################################################



"""

Proširenje (sc | hub_u | hub_d | mega) 
Dodato iz graspe embedding_randw + scripts/customrws: SCWalk, HubWalkUniform, HubWalkDistribution;
sintetičke labele (--label-mode); mega = prosek DeepWalk + UnbiasedWalk + SCWalk.

**DeepWalk** (NetworkX + W2V), 
**UnbiasedWalk**, 
plus iz graspe **SCWalk** i **HubWalk** 
(Uniform/Distribution, kao `scripts/customrws/`) 
i **`mega`** (prosek DW+UB+SC).

Načini (--method): deepwalk | unbiased | ensemble | sc | hub_u | hub_d | mega

Zahtevi: numpy, pandas, networkx, gensim, torch (kroz embedding_randw), klon third_party/graspe.

DeepWalk korpus preko NetworkX + Word2Vec (skip-gram) umesto pip paketa `deepwalk`;
UnbiasedWalk iz `embedding_randw`; ensemble = L2-norm pa prosek vektora; patch RWEmbBase za gensim.

DeepWalk NX + UnbiasedWalk (podrazumevano ensemble)

DeepWalk (embedding_deepwalk.py) 
i jednostavan RandW (UnbiasedWalk iz embedding_randw.py), 
plus opcija ensemble (prosek embeddinga).

Paket deepwalk nije u okruženju; 
implementiramo DeepWalk korpus preko NetworkX 
(isti algoritam kao u graspe-u) + UnbiasedWalk iz graspe embedding_randw.py.



Šta radi:

DeepWalk – korpus slučajnih šetnji na neusmerenom grafu (NetworkX), 
zatim Word2Vec (skip-gram), isti princip kao u graspe-u. 
Zvanični DeepWalkEmbedding traži pip paket deepwalk koji nije instaliran, 
pa je korpus ovde u NX (bez novog paketa).

UnbiasedWalk – iz embeddings/embedding_randw.py 
(graspe RandW: ravnomerna slučajna šetnja).

ensemble (podrazumevano) – L2-normalizacija oba embeddinga, 
zatim aritmetički prosek po čvoru → jedan vektor za pair_scores.

Patch
RWEmbBase.embed → epochs=TUNED_W2V_EPOCHS.
TypeIs stub (zbog torch preko lid_eval pri importu embedding_randw).
np.int, DGL stub.

"""
