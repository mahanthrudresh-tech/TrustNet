"""Simple, self-contained TrustNet backend.

Pipeline: create/load transaction CSV -> construct matrices -> calculate six signals
-> compute account risk scores -> detect directed cyclic groups -> save graph/reports.

Run demo:       python trustnet_backend.py
Run real CSV:   python trustnet_backend.py --csv transactions.csv
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

WEIGHTS = {"cycle": .25, "svd": .20, "qr": .15, "lstsq": .15, "spectral": .15, "cov": .10}
MIN_RING = 3
MAX_ACCOUNTS = 1000
COMPONENT_LABELS = {
    "cycle": "Cycle / matrix-power",
    "svd": "SVD reconstruction error",
    "qr": "QR projection residual",
    "lstsq": "Least-squares residual",
    "spectral": "Spectral structural risk",
    "cov": "Covariance anomaly",
}
SCEN = {"easy": 100, "moderate": 100, "stealth": 100, "repeat_stealth": 100, "many_rings": 200, "long_rings": 300, "varied_amounts": 200, "no_ring": 100, "decoys": 100}


def normalize_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Accept sender/receiver/amount (or source/target/value) and validate it."""
    if data.empty:
        raise ValueError("The transaction CSV is empty.")
    names = {str(c).strip().lower(): c for c in data.columns}
    aliases = {
        "sender": ("sender", "source", "from", "nameorig"),
        "receiver": ("receiver", "target", "to", "namedest"),
        "amount": ("amount", "value", "transaction_amount", "amt"),
    }
    selected = {}
    for target, options in aliases.items():
        selected[target] = next((names[x] for x in options if x in names), None)
        if selected[target] is None:
            raise ValueError(f"Missing {target} column. Accepted: {', '.join(options)}")
    df = pd.DataFrame({
        "sender": data[selected["sender"]],
        "receiver": data[selected["receiver"]],
        "amount": pd.to_numeric(data[selected["amount"]], errors="coerce"),
    })
    bad = df.sender.isna() | df.receiver.isna() | df.amount.isna() | (df.amount <= 0) | ~np.isfinite(df.amount.fillna(0))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have missing IDs or invalid amounts.")
    df["sender"] = df.sender.astype(str).str.strip()
    df["receiver"] = df.receiver.astype(str).str.strip()
    if (df.sender.eq("") | df.receiver.eq("")).any():
        raise ValueError("Account IDs cannot be blank.")
    self_transfers = df.sender.eq(df.receiver)
    removed = int(self_transfers.sum())
    df = df.loc[~self_transfers].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable transactions remain after removing self-transfers.")
    df.insert(0, "transaction_id", [f"TXN-{i:06d}" for i in range(1, len(df) + 1)])
    df.attrs["self_transfers_removed"] = removed
    return df


def make_demo_transactions(n_accounts=100, n_transactions=1000, seed=42, ring_sizes=None, scenario="easy"):
    """Create a controlled synthetic dataset and return (data, known planted rings).

    Background edges are a directed acyclic flow across three layers. Rings are directed
    cycles. Ring amounts use the same log-normal distribution as normal amounts for stealth
    scenarios; repeat_stealth repeats cycle transfers without making their amounts large.
    This is a test fixture, not a realistic model of every payment network.
    """
    if scenario not in SCEN:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    n_accounts = int(n_accounts or SCEN[scenario])
    n_transactions = int(n_transactions or 10 * n_accounts)
    if n_accounts < 12:
        raise ValueError("Use at least 12 accounts.")
    normal_amount = lambda: float(rng.lognormal(5.3, .5))
    if ring_sizes is None:
        if scenario in ("easy", "moderate", "stealth", "repeat_stealth", "decoys"):
            ring_sizes = [int(rng.integers(4, 9))]
        elif scenario == "many_rings":
            ring_sizes = rng.integers(3, 9, size=int(rng.integers(2, 6))).tolist()
        elif scenario == "long_rings":
            ring_sizes = [12, 25]
        elif scenario == "varied_amounts":
            ring_sizes = rng.integers(3, 10, size=3).tolist()
        else:
            ring_sizes = []
    ring_sizes = [int(k) for k in ring_sizes]
    if any(k < MIN_RING for k in ring_sizes) or sum(ring_sizes) > n_accounts - 6:
        raise ValueError("Each ring needs at least 3 accounts and enough normal accounts must remain.")

    ids = [f"ACC{i:03d}" for i in range(n_accounts)]
    shuffled = rng.permutation(n_accounts).tolist()
    groups, pos = [], 0
    for size in ring_sizes:
        groups.append(shuffled[pos:pos + size]); pos += size
    ring_nodes = {i for group in groups for i in group}
    normal_nodes = [i for i in range(n_accounts) if i not in ring_nodes]
    rng.shuffle(normal_nodes)
    rows = []

    for group in groups:
        repeats = 5 if scenario == "repeat_stealth" else 1
        if scenario == "moderate":
            amount_fn = lambda: float(rng.uniform(900, 1800))
        elif scenario == "varied_amounts":
            scale = float(np.exp(rng.uniform(np.log(700), np.log(6000))))
            amount_fn = lambda scale=scale: float(rng.uniform(0.8, 1.2) * scale)
        else:
            amount_fn = normal_amount
        for i, sender in enumerate(group):
            receiver = group[(i + 1) % len(group)]
            for _ in range(repeats):
                rows.append((sender, receiver, amount_fn()))

    layers = [list(map(int, layer)) for layer in np.array_split(normal_nodes, 3)]
    possible_pairs = [(s, r) for i in range(3) for j in range(i + 1, 3)
                      for s in layers[i] for r in layers[j] if s != r]
    if not possible_pairs:
        raise ValueError("Not enough normal accounts to create background traffic.")

    # Benign circles are intentionally added in the decoy scenario to exercise false positives.
    if scenario == "decoys" and len(normal_nodes) >= 9:
        for _ in range(3):
            cycle = rng.choice(normal_nodes, size=3, replace=False).tolist()
            for i, sender in enumerate(cycle):
                rows.append((int(sender), int(cycle[(i + 1) % 3]), normal_amount()))

    target = max(n_transactions, len(rows))
    while len(rows) < target:
        s, r = possible_pairs[int(rng.integers(0, len(possible_pairs)))]
        rows.append((s, r, normal_amount()))

    df = pd.DataFrame({"sender": [ids[s] for s, _, _ in rows],
                       "receiver": [ids[r] for _, r, _ in rows],
                       "amount": [float(a) for _, _, a in rows]})
    df.insert(0, "transaction_id", [f"TXN-{i:06d}" for i in range(1, len(df) + 1)])
    truth = [[ids[i] for i in group] for group in groups]
    return df, truth


def _robust_unit(values):
    x = np.asarray(values, dtype=float)
    median = np.median(x)
    scale = max(1.4826 * np.median(np.abs(x - median)), .1 * np.std(x), 1e-12)
    z = np.maximum((x - median) / scale, 0)
    return 1 - np.exp(-z / 5)


def analyze_transactions(data: pd.DataFrame, weights=None, max_accounts=MAX_ACCOUNTS):
    """Build matrices, compute six signals, score accounts, and extract cyclic groups."""
    transactions = normalize_transactions(data)
    removed_self = int(transactions.attrs.get("self_transfers_removed", 0))
    account_ids = list(dict.fromkeys(list(transactions.sender) + list(transactions.receiver)))
    n = len(account_ids)
    if n < 8:
        raise ValueError("At least 8 unique accounts are needed for covariance and projection features.")
    if n > max_accounts:
        raise ValueError(f"Found {n:,} accounts; the current dense-matrix cap is {max_accounts:,}. Use sparse methods for larger graphs.")
    index = {account: i for i, account in enumerate(account_ids)}
    tx = [(index[r.sender], index[r.receiver], float(r.amount)) for r in transactions.itertuples(index=False)]

    # A = total sent amount; B = edge presence; F = transaction frequency per directed pair.
    A = np.zeros((n, n)); F = np.zeros((n, n))
    for s, r, amount in tx:
        A[s, r] += amount; F[s, r] += 1
    B = (A > 0).astype(float)

    # Find cycle groups from directed strongly connected components.
    graph = nx.DiGraph(); graph.add_nodes_from(range(n)); graph.add_edges_from(zip(*np.where(B > 0)))
    sccs = [sorted(map(int, c)) for c in nx.strongly_connected_components(graph)]
    cycle_raw = np.zeros(n)
    for group in sccs:
        if len(group) >= 2:
            cycle_raw[group] = np.log1p(len(group) - 1)
    # Matrix power B^3 detects closed three-step walks for moderate-sized datasets.
    triangles = np.diag(B @ B @ B) if n <= 400 else np.zeros(n)
    repeat_total = np.maximum(F.sum(1) - B.sum(1), 0) + np.maximum(F.sum(0) - B.sum(0), 0)
    cycle_raw += .15 * np.log1p(triangles) + .15 * np.log1p(repeat_total)

    sent, received = A.sum(1), A.sum(0)
    out_degree, in_degree = B.sum(1), B.sum(0)
    out_count, in_count = F.sum(1), F.sum(0)
    average = (sent + received) / np.maximum(out_degree + in_degree, 1)
    balance = 1 - np.abs(sent - received) / (sent + received + 1)
    max_edge = np.maximum(A.max(1), A.max(0))
    X = np.column_stack([np.log1p(sent), np.log1p(received), np.log1p(average), out_degree, in_degree,
                         balance, np.log1p(max_edge), np.log1p(out_count), np.log1p(in_count),
                         np.log1p(np.maximum(out_count - out_degree, 0)),
                         np.log1p(np.maximum(in_count - in_degree, 0)), np.log1p(np.maximum(F.max(1), F.max(0)))])
    med = np.median(X, axis=0); mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    Z = (X - med) / np.maximum(np.maximum(mad, .1 * X.std(0)), 1e-9)

    # 2. SVD: reconstruction error of outgoing + incoming profiles.
    median_amount = max(float(np.median(A[A > 0])), 1e-12)
    Wp = B + .5 * np.log1p(A / median_amount) + .35 * np.log1p(F)
    P = np.hstack([Wp, Wp.T]); norms = np.linalg.norm(P, axis=1)
    bulk = P[norms <= np.quantile(norms, .8)]
    if len(bulk) < 3: bulk = P
    _, singular, vt = np.linalg.svd(bulk, full_matrices=False)
    total_energy = max(float(np.sum(singular ** 2)), 1e-12)
    rank = max(1, min(len(singular), int(np.searchsorted(np.cumsum(singular ** 2) / total_energy, .60) + 1)))
    V = vt[:rank].T; svd_raw = np.linalg.norm(P - (P @ V) @ V.T, axis=1)

    # 3. QR: projection residual from typical account vectors.
    ref = np.argsort(np.linalg.norm(Z, axis=1))[:min(6, n)]
    Q, _ = np.linalg.qr(Z[ref].T, mode="reduced")
    qr_raw = np.linalg.norm(Z - (Z @ Q) @ Q.T, axis=1)

    # 4. Least squares: amount residuals against sender/receiver activity and pair frequency.
    M = np.asarray([[1.0, out_degree[s], in_degree[r], np.log1p(F[s, r])] for s, r, _ in tx])
    y = np.asarray([a for _, _, a in tx], dtype=float)
    coefficients = np.linalg.lstsq(M, y, rcond=None)[0]; residual = y - M @ coefficients
    rmed = np.median(residual); rmad = max(1.4826 * np.median(np.abs(residual - rmed)), 1e-9)
    rz = (residual - rmed) / rmad; ls_raw = np.zeros(n); counts = np.zeros(n)
    for (s, r, _), value in zip(tx, rz):
        ls_raw[s] += max(float(value), 0); ls_raw[r] += max(float(value), 0); counts[s] += 1; counts[r] += 1
    ls_raw = np.divide(ls_raw, counts, out=np.zeros(n), where=counts > 0)

    # 5. Spectral: eigenvalues/eigenvectors of symmetrized profile matrix.
    eigenvalues, eigenvectors = np.linalg.eigh((Wp + Wp.T) / 2)
    magnitudes = np.abs(eigenvalues); emed = np.median(magnitudes)
    escale = max(1.4826 * np.median(np.abs(magnitudes - emed)), .1 * magnitudes.std(), 1e-12)
    outliers = magnitudes > emed + 3 * escale
    spectral_raw = (eigenvectors[:, outliers] ** 2) @ (eigenvalues[outliers] ** 2) if np.any(outliers) else np.zeros(n)

    # 6. Covariance: regularized Mahalanobis-style distance.
    znorm = np.linalg.norm(Z, axis=1); typical = Z[znorm <= np.quantile(znorm, .8)]
    if len(typical) < Z.shape[1] + 2: typical = Z
    cov = np.atleast_2d(np.cov(typical.T)); ridge = max(1e-6, 1e-3 * float(np.trace(cov)) / max(len(cov), 1))
    cov = cov + ridge * np.eye(len(cov)); delta = Z - typical.mean(0)
    cov_raw = np.sqrt(np.maximum(np.einsum("ij,ji->i", delta, np.linalg.solve(cov, delta.T)), 0))

    raw = {"cycle": cycle_raw, "svd": svd_raw, "qr": qr_raw, "lstsq": ls_raw, "spectral": spectral_raw, "cov": cov_raw}
    unit = {key: _robust_unit(value) for key, value in raw.items()}
    selected_weights = dict(WEIGHTS if weights is None else weights)
    if set(selected_weights) != set(WEIGHTS) or any(not np.isfinite(v) or v < 0 for v in selected_weights.values()):
        raise ValueError("Weights must be finite, non-negative, and include all six component keys.")
    weight_sum = sum(selected_weights.values())
    if weight_sum <= 0: raise ValueError("At least one component weight must be positive.")
    risk = sum(selected_weights[k] * unit[k] for k in WEIGHTS) / weight_sum

    # Ring candidates = strongly connected groups of 3+ accounts. Oversized SCCs are skipped.
    max_ring_size = max(MIN_RING, int(.35 * n)); groups = []; oversized = 0
    for group in sccs:
        if len(group) < MIN_RING: continue
        if len(group) > max_ring_size: oversized += 1; continue
        groups.append(group)
    groups.sort(key=lambda g: (min(g), len(g)))
    memberships = {}; ring_rows = []
    for number, group in enumerate(groups, start=1):
        ring_id = f"RING-{number:03d}"; members = [account_ids[i] for i in group]
        for i in group: memberships.setdefault(i, []).append(ring_id)
        inside = transactions[transactions.sender.isin(members) & transactions.receiver.isin(members)]
        ring_rows.append({"ring_id": ring_id, "account_count": len(members), "accounts": members,
                          "transaction_count": int(len(inside)), "internal_amount_total": float(inside.amount.sum()),
                          "mean_member_risk": float(np.mean(risk[group])), "cycle_signal_mean": float(np.mean(unit["cycle"][group]))})

    account_rows = []
    for i, account_id in enumerate(account_ids):
        score = float(risk[i])
        band = "Critical" if score >= .8 else "High" if score >= .6 else "Moderate" if score >= .35 else "Low"
        row = {"account_id": account_id, "risk_score": score, "risk_band": band,
               "flagged": any(i in group for group in groups), "ring_ids": ", ".join(memberships.get(i, []))}
        for key in WEIGHTS:
            row[f"{key}_score"] = float(unit[key][i]); row[f"{key}_raw"] = float(raw[key][i])
            row[f"{key}_contribution"] = float(selected_weights[key] * unit[key][i] / weight_sum)
        account_rows.append(row)
    accounts = pd.DataFrame(account_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)

    edges = transactions.groupby(["sender", "receiver"], as_index=False).agg(amount_total=("amount", "sum"), transaction_count=("amount", "size"))
    member_lookup = {name: set(memberships.get(i, [])) for i, name in enumerate(account_ids)}
    edges["shared_ring_ids"] = edges.apply(lambda r: ", ".join(sorted(member_lookup[r.sender] & member_lookup[r.receiver])), axis=1)
    edges["is_ring_edge"] = edges.shared_ring_ids.ne("")
    components = pd.DataFrame([{
        "component": key, "label": COMPONENT_LABELS[key], "description": COMPONENT_LABELS[key],
        "weight": selected_weights[key] / weight_sum, "weight_percent": 100 * selected_weights[key] / weight_sum,
        "mean_score": float(unit[key].mean()), "max_score": float(unit[key].max()),
        "mean_weighted_contribution": float((unit[key] * selected_weights[key] / weight_sum).mean()),
    } for key in WEIGHTS])

    diagnostics = {
        "rank_A": int(np.linalg.matrix_rank(A)), "svd_rank": int(rank), "outlier_eigs": int(outliers.sum()),
        "cov_min_eig": float(np.linalg.eigvalsh(cov).min()),
        "Q_orthonormal": bool(np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-6)),
        "lstsq_vs_numpy": 0.0, "self_transfers_removed": removed_self,
        "repeated_directed_pairs": int(np.sum(F > 1)), "max_transactions_per_pair": int(F.max()),
        "oversized_sccs_ignored": oversized, "dense_matrix_warning": n >= 500,
        "risk_is_probability": False,
    }
    metrics = {"account_count": n, "transaction_count": len(transactions), "unique_directed_edges": len(edges),
               "flagged_account_count": int(accounts.flagged.sum()), "candidate_ring_count": len(ring_rows),
               "high_risk_account_count": int((risk >= .6).sum()), "critical_account_count": int((risk >= .8).sum()),
               "transaction_amount_total": float(transactions.amount.sum()), "transaction_amount_median": float(transactions.amount.median()),
               "self_transfers_removed": removed_self}
    engine_result = {"A": A, "B": B, "F": F, "raw": raw, "unit": unit, "risk": risk, "ring_indices": groups}
    return {"accounts": accounts,
            "rings": pd.DataFrame(ring_rows, columns=["ring_id", "account_count", "accounts", "transaction_count", "internal_amount_total", "mean_member_risk", "cycle_signal_mean"]),
            "transactions": transactions, "edges": edges, "components": components, "metrics": metrics,
            "diagnostics": diagnostics, "weights": {k: selected_weights[k] / weight_sum for k in WEIGHTS},
            "engine_result": engine_result, "account_ids": account_ids}


def save_graph(analysis, path):
    """Save the directed network graph with detected candidate rings in red."""
    accounts, edges = analysis["accounts"], analysis["edges"]
    ring_members = accounts.loc[accounts.flagged, "account_id"].tolist()
    chosen = list(dict.fromkeys(ring_members))
    for account in accounts.account_id:
        if account not in chosen and len(chosen) < 160: chosen.append(account)
    chosen = chosen[:160]; chosen_set = set(chosen)
    e = edges[edges.sender.isin(chosen_set) & edges.receiver.isin(chosen_set)]
    red = e[e.is_ring_edge]; grey = e[~e.is_ring_edge].nlargest(800, "amount_total")
    G = nx.DiGraph(); G.add_nodes_from(chosen); G.add_edges_from(zip(pd.concat([red, grey]).sender, pd.concat([red, grey]).receiver))
    pos = nx.spring_layout(G, seed=42, iterations=80)
    lookup = accounts.set_index("account_id").to_dict("index")
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0b1120"); ax.set_facecolor("#0b1120")
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(grey.sender, grey.receiver)), edge_color="#718096", alpha=.2, width=.5, arrows=False, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(red.sender, red.receiver)), edge_color="#ef4444", width=2, arrows=True, arrowsize=12, ax=ax)
    normal = [x for x in G if x not in set(ring_members)]; flagged = [x for x in G if x in set(ring_members)]
    nx.draw_networkx_nodes(G, pos, nodelist=normal, node_color="#8b9bb0", node_size=[30 + 120*lookup[x]["risk_score"] for x in normal], ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=flagged, node_color="#ef4444", node_size=220, edgecolors="white", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={x: x for x in flagged}, font_color="white", font_size=7, ax=ax)
    ax.set_title(f"TrustNet · {analysis['metrics']['candidate_ring_count']} candidate ring(s) · red = ring candidates", color="white")
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)


def export_outputs(analysis, out_dir="."):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    analysis["transactions"].to_csv(out / "transaction_history.csv", index=False)
    analysis["accounts"].to_csv(out / "trustnet_account_risk.csv", index=False)
    analysis["components"].to_csv(out / "trustnet_component_scores.csv", index=False)
    analysis["edges"].to_csv(out / "trustnet_graph_edges.csv", index=False)
    rings = analysis["rings"].copy()
    if not rings.empty: rings["accounts"] = rings.accounts.map(lambda x: ", ".join(x))
    rings.to_csv(out / "trustnet_detected_rings.csv", index=False)
    with open(out / "trustnet_diagnostics.json", "w", encoding="utf-8") as f: json.dump(analysis["diagnostics"], f, indent=2)
    save_graph(analysis, out / "trustnet_graph.png")
    comp = analysis["components"].sort_values("weight_percent")
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0b1120"); ax.set_facecolor("#0b1120")
    bars = ax.barh(comp.label, comp.weight_percent, color="#e63946"); ax.tick_params(colors="#cbd5e1")
    ax.set_title("TrustNet component weights", color="white"); ax.set_xlabel("Weight (%)", color="#cbd5e1")
    for bar, value in zip(bars, comp.weight_percent): ax.text(value + .5, bar.get_y()+bar.get_height()/2, f"{value:.0f}%", color="white", va="center")
    ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig(out / "trustnet_component_breakdown.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)



def evaluate_scenarios(runs=3, scenarios=None):
    """Small repeatable synthetic benchmark for the later dashboard page."""
    if not 1 <= int(runs) <= 20:
        raise ValueError("runs must be between 1 and 20.")
    scenarios = scenarios or ["easy", "stealth", "decoys", "no_ring"]
    rows = []
    corr_values = []

    for scenario in scenarios:
        if scenario not in SCEN:
            raise ValueError(f"Unknown scenario: {scenario}")
        true_account_count = detected_account_count = account_hits = 0
        true_ring_count = detected_ring_count = ring_hits = ring_precision_hits = 0
        false_alarm_runs = 0

        for seed in range(int(runs)):
            data, truth = make_demo_transactions(scenario=scenario, seed=42 + seed)
            analysis = analyze_transactions(data)
            detected = analysis["rings"]["accounts"].tolist() if not analysis["rings"].empty else []
            truth_sets = [set(group) for group in truth]
            detected_sets = [set(group) for group in detected]
            pairs = []
            for di, ds in enumerate(detected_sets):
                for ti, ts in enumerate(truth_sets):
                    union = ds | ts
                    overlap = len(ds & ts) / len(union) if union else 0.0
                    if overlap >= 0.70:
                        pairs.append((overlap, di, ti))
            used_d, used_t = set(), set()
            for _, di, ti in sorted(pairs, reverse=True):
                if di not in used_d and ti not in used_t:
                    used_d.add(di); used_t.add(ti)
            ring_hits += len(used_t)
            ring_precision_hits += len(used_d)
            true_ring_count += len(truth_sets)
            detected_ring_count += len(detected_sets)

            planted = {account for group in truth_sets for account in group}
            flagged = set(analysis["accounts"].loc[analysis["accounts"]["flagged"], "account_id"])
            account_hits += len(planted & flagged)
            true_account_count += len(planted)
            detected_account_count += len(flagged)
            if scenario == "no_ring" and detected_sets:
                false_alarm_runs += 1

            score_matrix = np.column_stack([
                analysis["engine_result"]["unit"][key] for key in WEIGHTS
            ])
            if np.all(score_matrix.std(axis=0) > 0):
                corr_values.append(np.corrcoef(score_matrix, rowvar=False))

        rows.append({
            "scenario": scenario,
            "method": "TrustNet-6 (six matrix components)",
            "account_recall": account_hits / true_account_count if true_account_count else np.nan,
            "account_precision": account_hits / detected_account_count if detected_account_count else (1.0 if scenario == "no_ring" else 0.0),
            "ring_recall": ring_hits / true_ring_count if true_ring_count else np.nan,
            "ring_precision": ring_precision_hits / detected_ring_count if detected_ring_count else (1.0 if scenario == "no_ring" else 0.0),
            "false_alarm_runs": false_alarm_runs if scenario == "no_ring" else np.nan,
            "runs": int(runs),
        })

    correlation = np.mean(corr_values, axis=0) if corr_values else np.eye(len(WEIGHTS))
    correlation_df = pd.DataFrame(correlation, columns=list(WEIGHTS))
    correlation_df.insert(0, "component", list(WEIGHTS))
    return {
        "metrics": pd.DataFrame(rows),
        "correlation": correlation_df,
        "runs": int(runs),
        "scenarios": scenarios,
        "methods": ["TrustNet-6 (six matrix components)"],
    }


def run_pipeline(csv_path=None, n_accounts=100, n_transactions=1000, seed=42, scenario="easy", ring_sizes=None, out_dir="."):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print("[1/5] Generate/load transaction history CSV")
    if csv_path:
        data = normalize_transactions(pd.read_csv(csv_path)); truth = None
    else:
        data, truth = make_demo_transactions(n_accounts, n_transactions, seed, ring_sizes, scenario)
    data.to_csv(out / "transaction_history.csv", index=False)
    print(f"      Saved {len(data):,} transaction rows")
    print("[2/5] Build amount, presence and frequency matrices; run SVD, QR, least squares, eigen and covariance operations")
    start = time.time(); result = analyze_transactions(data); print(f"      Done in {time.time()-start:.2f}s")
    print("[3/5] Calculate weighted risk scores")
    print("[4/5] Find strongly connected groups that may be fraud rings")
    print(f"      Found {result['metrics']['candidate_ring_count']} candidate ring(s)")
    print("[5/5] Save graph and result files")
    export_outputs(result, out)
    print("      transaction_history.csv")
    print("      trustnet_account_risk.csv")
    print("      trustnet_component_scores.csv")
    print("      trustnet_graph_edges.csv")
    print("      trustnet_detected_rings.csv")
    print("      trustnet_graph.png")
    print("      trustnet_component_breakdown.png")
    print("      trustnet_diagnostics.json")
    print("\nTop 10 accounts:")
    print(result["accounts"][["account_id", "risk_score", "risk_band", "flagged", "ring_ids"]].head(10).to_string(index=False))
    if truth is not None:
        detected = result["rings"].accounts.tolist() if not result["rings"].empty else []
        matches = sum(any(set(d) == set(t) for d in detected) for t in truth)
        print(f"\nSynthetic check: {matches}/{len(truth)} planted ring(s) matched (not real-world validation).")
    print("Risk scores are investigative rankings; a detected ring is a candidate, not proof of fraud.")
    return result


def main():
    parser = argparse.ArgumentParser(description="TrustNet standalone backend")
    parser.add_argument("--csv", help="Input an existing CSV instead of generating demo data")
    parser.add_argument("--accounts", type=int, default=100)
    parser.add_argument("--transactions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=list(SCEN), default="easy")
    parser.add_argument("--ring-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    run_pipeline(args.csv, args.accounts, args.transactions, args.seed, args.scenario, args.ring_sizes, args.out_dir)


if __name__ == "__main__":
    main()

