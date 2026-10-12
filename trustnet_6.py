from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

SCEN = {
    "easy": {"N": 100},
    "many_rings": {"N": 200},
    "long_rings": {"N": 300},
    "varied_amounts": {"N": 200},
    "moderate": {"N": 100},
    "stealth": {"N": 100},
    "repeat_stealth": {"N": 100},
    "decoys": {"N": 100},
    "no_ring": {"N": 100},
}

def simulate(
    seed,
    scenario="easy",
    N=None,
    T=None,
    ring_sizes=None,
):
    if scenario not in SCEN:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)
    N = N if N is not None else SCEN[scenario]["N"]
    T = T if T is not None else 10 * N

    def normal():
        return float(rng.lognormal(5.3, 0.5))

    def big():
        return float(rng.uniform(3500, 5500))

    def moderate():
        return float(rng.uniform(900, 1800))

    if ring_sizes is None:
        if scenario in ("easy", "moderate", "stealth",
                        "repeat_stealth", "decoys"):
            ring_sizes = [5]
        elif scenario == "no_ring":
            ring_sizes = []
        elif scenario == "many_rings":
            ring_sizes = rng.integers(
                3, 12, size=int(rng.integers(2, 6))
            ).tolist()
        elif scenario == "long_rings":
            ring_sizes = [12, 25]
        elif scenario == "varied_amounts":
            ring_sizes = rng.integers(3, 10, size=3).tolist()
        else:
            ring_sizes = [5]

    ring_sizes = [int(k) for k in ring_sizes]

    if any(k < MIN_RING for k in ring_sizes):
        raise ValueError(
            f"Ring sizes must be at least {MIN_RING}."
        )

    if sum(ring_sizes) > N:
        raise ValueError("Ring sizes exceed the account count.")

    if scenario == "moderate":
        amount_fn = moderate
    elif scenario in ("stealth", "repeat_stealth"):
        amount_fn = normal
    else:
        amount_fn = big

    permutation = rng.permutation(N).tolist()
    position = 0
    tx = []
    rings = []

    for size in ring_sizes:
        ids = permutation[position:position + size]
        position += size

        ring_amount = amount_fn

        if scenario == "varied_amounts":
            scale = float(np.exp(
                rng.uniform(np.log(700), np.log(6000))
            ))
            ring_amount = (
                lambda scale=scale:
                float(rng.uniform(0.8, 1.2) * scale)
            )

        for i in range(size):
            sender = ids[i]
            receiver = ids[(i + 1) % size]

            tx.append((sender, receiver, ring_amount()))

            if scenario == "repeat_stealth":
                for _ in range(5):
                    tx.append((
                        sender,
                        receiver,
                        ring_amount(),
                    ))

        for _ in range(size // 3):
            sender, receiver = rng.choice(
                ids, size=2, replace=False
            )
            tx.append((
                int(sender),
                int(receiver),
                ring_amount(),
            ))

        tx.append((
            int(rng.choice(N)),
            ids[0],
            300.0,
        ))
        tx.append((
            ids[-1],
            int(rng.choice(N)),
            250.0,
        ))

        rings.append(sorted(ids))

    if scenario == "decoys":
        ring_accounts = {
            account for group in rings for account in group
        }
        others = [
            i for i in range(N)
            if i not in ring_accounts
        ]

        for payer in rng.choice(
            others, size=min(5, len(others)), replace=False
        ):
            for _ in range(4):
                tx.append((
                    int(payer),
                    int(rng.choice(others)),
                    big(),
                ))

        if len(others) >= 9:
            for _ in range(3):
                a, b, c = rng.choice(
                    others, size=3, replace=False
                )
                tx.extend([
                    (int(a), int(b), normal()),
                    (int(b), int(c), normal()),
                    (int(c), int(a), normal()),
                ])

    while len(tx) < T:
        sender, receiver = rng.choice(
            N, size=2, replace=False
        )
        tx.append((
            int(sender),
            int(receiver),
            normal(),
        ))

    return tx, rings, N

WEIGHTS = {
    "cycle": 0.25,
    "svd": 0.20,
    "qr": 0.15,
    "lstsq": 0.15,
    "spectral": 0.15,
    "cov": 0.10,
}

CORE4 = {
    "cycle": 0.25,
    "svd": 0.20,
    "qr": 0.15,
    "lstsq": 0.15,
    "spectral": 0.0,
    "cov": 0.0,
}

MIN_RING = 3
LEVELS = 30
GATE_MAD = 3.0
MERGE_RATIO = 0.5
Z_CUT = 3.0
RISK_FLOOR = 0.10

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=150,
)


# ============================================================
# HELPERS
# ============================================================

def robust_scale(x):
    x = np.asarray(x, dtype=float)

    median = np.median(x)
    scale = max(
        1.4826 * np.median(np.abs(x - median)),
        0.1 * np.std(x),
        1e-12,
    )

    return median, scale


def to_unit(x, k=5.0):
    """Robustly map positive outlier evidence to a range below 1."""
    median, scale = robust_scale(x)
    z = np.maximum((x - median) / scale, 0.0)
    return 1.0 - np.exp(-z / k)


def gauss_solve(A, B):
    """Solve AX=B with Gaussian elimination and partial pivoting."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    vector_input = B.ndim == 1

    if vector_input:
        B = B[:, None]

    M = np.hstack([A.copy(), B.copy()])
    n = len(A)

    for c in range(n):
        p = c + np.argmax(np.abs(M[c:, c]))

        if abs(M[p, c]) < 1e-12:
            raise np.linalg.LinAlgError(
                "Singular matrix encountered during Gaussian elimination."
            )

        M[[c, p]] = M[[p, c]]

        if c + 1 < n:
            factors = M[c + 1:, c] / M[c, c]
            M[c + 1:] -= np.outer(factors, M[c])

    X = np.zeros((n, M.shape[1] - n))

    for i in range(n - 1, -1, -1):
        X[i] = (
            M[i, n:]
            - M[i, i + 1:n] @ X[i + 1:n]
        ) / M[i, i]

    return X[:, 0] if vector_input else X


# ============================================================
# ACCOUNT FEATURES
# ============================================================

def account_features(A, B, F=None):
    """
    Build account features from transaction amounts, connectivity,
    flow balance and transaction frequency.
    """
    sent = A.sum(axis=1)
    received = A.sum(axis=0)

    out_degree = B.sum(axis=1)
    in_degree = B.sum(axis=0)

    average_amount = (
        (sent + received)
        / np.maximum(out_degree + in_degree, 1)
    )

    balance = (
        1.0
        - np.abs(sent - received)
        / (sent + received + 1.0)
    )

    max_edge = np.maximum(
        A.max(axis=1),
        A.max(axis=0),
    )

    if F is None:
        F = B.copy()

    tx_out = F.sum(axis=1)
    tx_in = F.sum(axis=0)

    repeat_out = np.maximum(tx_out - out_degree, 0)
    repeat_in = np.maximum(tx_in - in_degree, 0)

    max_repeat = np.maximum(
        F.max(axis=1),
        F.max(axis=0),
    )

    X = np.column_stack([
        np.log1p(sent),
        np.log1p(received),
        np.log1p(average_amount),
        out_degree,
        in_degree,
        balance,
        np.log1p(max_edge),
        np.log1p(tx_out),
        np.log1p(tx_in),
        np.log1p(repeat_out),
        np.log1p(repeat_in),
        np.log1p(max_repeat),
    ])

    median = np.median(X, axis=0)
    mad = 1.4826 * np.median(
        np.abs(X - median),
        axis=0,
    )

    scale = np.maximum(
        np.maximum(mad, 0.1 * X.std(axis=0)),
        1e-9,
    )

    return (X - median) / scale


# ============================================================
# 1. CYCLE AND MATRIX-POWER ANALYSIS
# ============================================================

def mutual_reach(H):
    """
    Detect mutual directed reachability using Boolean matrix powers
    and repeated squaring.
    """
    H = np.asarray(H, dtype=np.float32)
    n = H.shape[0]

    R = (
        (H + np.eye(n, dtype=np.float32)) > 0
    ).astype(np.float32)

    for _ in range(
        int(np.ceil(np.log2(max(n, 2)))) + 1
    ):
        R2 = ((R @ R) > 0).astype(np.float32)

        if np.array_equal(R2, R):
            break

        R = R2

    return R * R.T


def cycle_strength(A, levels=LEVELS, min_ring=MIN_RING):
    """
    Measure the strongest transaction-amount threshold at which
    each account belongs to a directed cycle with enough members.
    """
    positive = A[A > 0]

    strength = np.zeros(len(A))

    if positive.size == 0:
        return strength

    thresholds = np.unique(
        np.quantile(
            positive,
            np.linspace(0, 1, levels),
        )
    )[::-1]

    for threshold in thresholds:
        H = (A >= threshold).astype(np.float32)
        C = mutual_reach(H)

        new = (
            (C.sum(axis=1) >= min_ring)
            & (strength == 0)
        )

        strength[new] = threshold

    return np.log1p(strength)


# ============================================================
# 2. SVD RECONSTRUCTION ERROR
# ============================================================

def svd_error(Wp, energy=0.60, trim=0.80):
    """Measure account-profile reconstruction error using SVD."""
    P = np.hstack([Wp, Wp.T])
    norms = np.linalg.norm(P, axis=1)

    bulk = norms <= np.quantile(norms, trim)
    typical = P[bulk]

    if typical.shape[0] < 2:
        typical = P

    _, S, Vt = np.linalg.svd(
        typical,
        full_matrices=False,
    )

    total_energy = np.sum(S ** 2)

    if total_energy <= 1e-12:
        return np.linalg.norm(P, axis=1), 1

    cumulative = np.cumsum(S ** 2) / total_energy

    rank = min(
        int(np.searchsorted(cumulative, energy) + 1),
        len(S),
    )

    Vr = Vt[:rank].T

    reconstructed = (P @ Vr) @ Vr.T
    residual = np.linalg.norm(
        P - reconstructed,
        axis=1,
    )

    return residual, rank


# ============================================================
# 3. QR PROJECTION RESIDUAL
# ============================================================

def qr_residual(Z, ref_size=3):
    """
    Use QR decomposition to create an orthonormal reference
    subspace and calculate each account's projection residual.
    """
    ref_size = min(ref_size, len(Z))

    reference = np.argsort(
        np.linalg.norm(Z, axis=1)
    )[:ref_size]

    Q, _ = np.linalg.qr(
        Z[reference].T,
        mode="reduced",
    )

    residual = np.linalg.norm(
        Z - (Z @ Q) @ Q.T,
        axis=1,
    )

    return residual, Q


# ============================================================
# 4. LEAST-SQUARES ANOMALY
# ============================================================

def lstsq_anomaly(tx, B, N):
    """Aggregate positive transaction prediction residuals by account."""
    out_degree = B.sum(axis=1)
    in_degree = B.sum(axis=0)

    M = np.array([
        [1.0, out_degree[s], in_degree[r]]
        for s, r, _ in tx
    ])

    y = np.array([
        amount for _, _, amount in tx
    ])

    if len(y) == 0:
        return np.zeros(N), 0.0

    # Small ridge term helps avoid singular normal equations.
    normal_matrix = M.T @ M
    normal_rhs = M.T @ y

    try:
        w = gauss_solve(
            normal_matrix + 1e-9 * np.eye(normal_matrix.shape[0]),
            normal_rhs,
        )
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(
            M,
            y,
            rcond=None,
        )[0]

    residual = y - M @ w

    median = np.median(residual)
    mad = 1.4826 * np.median(
        np.abs(residual - median)
    )

    z = (
        residual - median
    ) / max(mad, 1e-9)

    score = np.zeros(N)
    count = np.zeros(N)

    for (sender, receiver, _), value in zip(tx, z):
        for account in (sender, receiver):
            score[account] += max(float(value), 0.0)
            count[account] += 1

    score = np.divide(
        score,
        count,
        out=np.zeros(N),
        where=count > 0,
    )

  