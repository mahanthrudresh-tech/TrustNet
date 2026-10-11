"""
TrustNet (scalable version) - works for any number of accounts / transactions.
Change the three settings below, e.g. N_ACCOUNTS=100, N_TX=1000.
"""
import time
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# ======================= SETTINGS =======================
N_ACCOUNTS = 100
N_TX       = 1000
RING_SIZE  = 5
SEED       = 42
# ========================================================
np.set_printoptions(precision=3, suppress=True, linewidth=150)


def minmax(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


# ---------------- Stage 0: data (ring placed at RANDOM accounts) ----------------
def simulate(seed, N=N_ACCOUNTS, T=N_TX, ring_size=RING_SIZE):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    ring, normal = sorted(perm[:ring_size].tolist()), perm[ring_size:]
    r = ring
    tx = []
    for i in range(ring_size):                                   # ring: big circular transfers
        tx.append((r[i], r[(i + 1) % ring_size], float(rng.uniform(3500, 5500))))
    tx.append((r[1], r[3], float(rng.uniform(3500, 5500))))      # a chord
    tx.append((int(normal[0]), r[0], 300.0))                     # links to the normal world
    tx.append((r[2], int(normal[1]), 250.0))
    while len(tx) < T:                                           # normal: small random transfers
        s, d = rng.choice(normal, size=2, replace=False)
        tx.append((int(s), int(d), float(rng.lognormal(5.3, 0.5))))
    return tx, ring


# ---------------- Stage 4 helper: power iteration (replaces np.linalg.eig) -------
def power_iteration(P, d=0.85, iters=200, tol=1e-10):
    """Dominant eigenvector of the Google matrix G = d*P + (1-d)/n, WITHOUT forming G.
    Cost per step O(nnz) instead of O(n^3) for a full eig."""
    n = P.shape[0]
    x = np.ones(n) / n
    for _ in range(iters):
        x_new = d * (P @ x) + (1 - d) / n * x.sum()
        x_new /= x_new.sum()
        if np.abs(x_new - x).sum() < tol:
            break
        x = x_new
    return x_new


def pipeline(tx, N, ref_size=3):
    # ---- Stage 1: matrix ----
    A = np.zeros((N, N))
    for s, r, amt in tx:
        A[s, r] += amt
    B = (A > 0).astype(float)

    # "heavy" edges = amounts far above the typical amount (robust median + 3 MAD on log scale)
    la = np.log(A[A > 0])
    thr = np.exp(np.median(la) + 3 * 1.4826 * np.median(np.abs(la - np.median(la))))
    H = (A > thr).astype(float)

    # ---- Stage 2: Gram-Schmidt (QR) + projection ----
    out_deg, in_deg = B.sum(1), B.sum(0)
    sent, recv = A.sum(1), A.sum(0)
    avg_amt = (sent + recv) / np.maximum(out_deg + in_deg, 1)
    H3 = np.linalg.matrix_power(H, 3); H4 = H3 @ H; H5 = H4 @ H
    cyc = np.diag(H3) + np.diag(H4) + np.diag(H5)               # heavy-money cycles through node
    heavy_deg = H.sum(0) + H.sum(1)
    X = np.column_stack([np.log1p(sent), np.log1p(recv), np.log1p(avg_amt),
                         out_deg, in_deg, cyc, heavy_deg])
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0)
    mad = np.where(mad == 0, X.std(0) + 1e-9, mad)
    Z = (X - med) / mad
    ref = np.argsort(np.linalg.norm(Z, axis=1))[:ref_size]       # most typical accounts
    Q, _ = np.linalg.qr(Z[ref].T)
    residual = np.linalg.norm(Z - Z @ Q @ Q.T, axis=1)

    # ---- Stage 3: least squares (structure-only predictors) ----
    M = np.array([[1.0, out_deg[s], in_deg[r]] for s, r, _ in tx])
    y = np.array([amt for _, _, amt in tx])
    w, *_ = np.linalg.lstsq(M, y, rcond=None)
    e = y - M @ w
    z = (e - np.median(e)) / (np.median(np.abs(e - np.median(e))) + 1e-9)
    ls, cnt = np.zeros(N), np.zeros(N)
    for (s, r, _), zi in zip(tx, z):
        for a in (s, r):
            ls[a] += max(zi, 0); cnt[a] += 1
    ls = np.divide(ls, cnt, out=np.zeros(N), where=cnt > 0)

    # ---- Stage 4: eigenvector centrality (power iteration) ----
    P = A.T.copy()
    out_sum = A.sum(1)
    for i in range(N):
        P[:, i] = P[:, i] / out_sum[i] if out_sum[i] > 0 else 1.0 / N
    cent = power_iteration(P)

    # ---- Stage 5: risk + ring ----
    risk = 0.30 * minmax(residual) + 0.40 * minmax(ls) + 0.30 * minmax(cent)
    risk /= risk.max()
    # robust outlier cutoff (median + 3 MAD): stays strict however many accounts there are
    med_r = np.median(risk)
    cutoff = med_r + 3 * 1.4826 * np.median(np.abs(risk - med_r))
    cand = [i for i in range(N) if risk[i] > cutoff]
    sub = nx.DiGraph(); sub.add_nodes_from(cand)
    sub.add_edges_from((i, j) for i in cand for j in cand if A[i, j] > 0)
    sccs = [c for c in nx.strongly_connected_components(sub) if len(c) >= 3]
    flagged = sorted(max(sccs, key=lambda c: sum(risk[i] for i in c))) if sccs else []
    return dict(A=A, risk=risk, residual=residual, ls=ls, cent=cent, flagged=flagged)


def score(flagged, ring):
    tp = len(set(flagged) & set(ring)); fp = len(set(flagged) - set(ring))
    return tp / len(ring), (tp / (tp + fp) if tp + fp else 0.0)


if __name__ == "__main__":
    N = N_ACCOUNTS
    names = [f"ACC{i:03d}" for i in range(N)]
    tx, ring = simulate(SEED)
    t0 = time.time(); res = pipeline(tx, N); dt = time.time() - t0
    risk, flagged = res["risk"], res["flagged"]
    print(f"Accounts={N}  Transactions={len(tx)}  runtime={dt:.3f}s")

    print("\nTOP 10 RISKIEST ACCOUNTS")
    print(f"{'Rank':<5}{'Account':<9}{'Risk':>6}{'Proj.dev':>10}{'LS':>8}{'Central.':>10}  Flag")
    for rank, i in enumerate(np.argsort(-risk)[:10], 1):
        print(f"{rank:<5}{names[i]:<9}{risk[i]:>6.2f}{res['residual'][i]:>10.2f}"
              f"{res['ls'][i]:>8.2f}{res['cent'][i]:>10.4f}  {'FRAUD RING' if i in flagged else ''}")
    rec, prec = score(flagged, ring)
    print(f"\nFlagged: {[names[i] for i in flagged]}\nPlanted: {[names[i] for i in ring]}")
    print(f"Recall={rec:.0%}  Precision={prec:.0%}")

    ok = 0; runs = 50
    for s in range(runs):
        t_, r_ = simulate(s); a, b = score(pipeline(t_, N)["flagged"], r_)
        ok += (a == 1 and b == 1)
    print(f"\nRobustness: ring found perfectly in {ok}/{runs} random datasets")

    # ---- graph: only draw ring + top-risk accounts clearly, rest small/faint ----
    Gx = nx.DiGraph(); Gx.add_nodes_from(range(N))
    for s, r, amt in tx:
        Gx.add_edge(s, r, weight=Gx[s][r]["weight"] + amt if Gx.has_edge(s, r) else amt)
    others = [i for i in range(N) if i not in flagged]
    pos = nx.spring_layout(Gx.subgraph(others), seed=1, k=0.35, iterations=100)
    pos = {k: np.array(v) * 1.5 + np.array([-0.6, 0]) for k, v in pos.items()}
    if flagged:
        pos.update(nx.circular_layout(Gx.subgraph(flagged), scale=0.6, center=(2.4, 0)))
    col = ["#e63946" if i in flagged else "#b0b7c3" for i in range(N)]
    siz = [320 if i in flagged else 40 + 150 * risk[i] for i in range(N)]
    heavy = [(s, r) for s, r in Gx.edges() if s in flagged and r in flagged]
    light = [e for e in Gx.edges() if e not in heavy]
    plt.figure(figsize=(11, 7.5))
    nx.draw_networkx_edges(Gx, pos, edgelist=light, edge_color="#dde1e7", width=0.4, arrows=False)
    nx.draw_networkx_edges(Gx, pos, edgelist=heavy, edge_color="#e63946", width=2.5, node_size=siz,
                           connectionstyle="arc3,rad=0.1", min_source_margin=8, min_target_margin=8)
    nx.draw_networkx_nodes(Gx, pos, node_color=col, node_size=siz, edgecolors="white")
    nx.draw_networkx_labels(Gx, pos, labels={i: names[i][-3:] for i in flagged}, font_size=8)
    plt.title(f"TrustNet - {N} accounts, {len(tx)} transactions\n(red = flagged ring)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig("trustnet_scalable_graph.png", dpi=180)
    print("Saved trustnet_scalable_graph.png")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
