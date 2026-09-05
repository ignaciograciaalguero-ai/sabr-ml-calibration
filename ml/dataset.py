"""
ml/dataset.py
=============
Generación del dataset sintético para entrenar el surrogate model SABR.

Cada muestra es un par (inputs, target):
    inputs  = (alpha, beta, rho, nu, F, moneyness, T)   shape (7,)
    target  = sigma_B  (volatilidad implícita Black-76)  shape (1,)

Uso
---
    python ml/dataset.py                   # genera 200k muestras
    python ml/dataset.py --n 500000        # dataset más grande
    python ml/dataset.py --test-only       # test rápido con 10k
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


from sabr.pricer import hagan_vol




# ---------------------------------------------------------------------------
# Rangos del espacio de parámetros
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "alpha":     (0.001, 0.50),
    "beta":      [0.0, 0.25, 0.5, 0.75, 1.0],
    "rho":       (-0.99, 0.99),
    "nu":        (0.01, 2.00),
    "F":         (0.001, 0.15),
    "moneyness": (0.50, 2.00),
    "T":         (0.25, 30.00),
}

INPUT_COLS = ["alpha", "beta", "rho", "nu", "F", "moneyness", "T"]
TARGET_COL = "sigma_B"
N_INPUTS   = len(INPUT_COLS)


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
def sample_parameters(n_samples: int, rng: np.random.Generator) -> dict:
    r = PARAM_RANGES

    def log_uniform(lo, hi, n):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    return {
        "alpha":     log_uniform(r["alpha"][0],     r["alpha"][1],     n_samples),
        "beta":      rng.choice(r["beta"],           size=n_samples),
        "rho":       rng.uniform(r["rho"][0],        r["rho"][1],       n_samples),
        "nu":        log_uniform(r["nu"][0],         r["nu"][1],        n_samples),
        "F":         log_uniform(r["F"][0],          r["F"][1],         n_samples),
        "moneyness": log_uniform(r["moneyness"][0],  r["moneyness"][1], n_samples),
        "T":         log_uniform(r["T"][0],          r["T"][1],         n_samples),
    }


# ---------------------------------------------------------------------------
# Evaluación y filtrado
# ---------------------------------------------------------------------------
def evaluate_pricer(params: dict) -> np.ndarray:
    K = params["F"] * params["moneyness"]
    return hagan_vol(
        F=params["F"], K=K, T=params["T"],
        alpha=params["alpha"], beta=params["beta"],
        rho=params["rho"],     nu=params["nu"],
        validate=False,
    )


def filter_valid(
    params: dict, sigma_B: np.ndarray,
    vol_min: float = 1e-6, vol_max: float = 5.0,
) -> tuple[dict, np.ndarray]:
    mask = np.isfinite(sigma_B) & (sigma_B > vol_min) & (sigma_B < vol_max)
    return {k: v[mask] for k, v in params.items()}, sigma_B[mask]


def build_input_matrix(params: dict) -> np.ndarray:
    return np.column_stack([params[col] for col in INPUT_COLS])


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def compute_normalisation_stats(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Calcula media y std SOLO sobre el train set (evita data leakage)."""
    return {
        "X_mean": X_train.mean(axis=0),
        "X_std":  X_train.std(axis=0) + 1e-8,
        "y_mean": float(y_train.mean()),
        "y_std":  float(y_train.std()) + 1e-8,
    }


# ---------------------------------------------------------------------------
# Guardado y carga
# ---------------------------------------------------------------------------
def save_dataset(
    X_train, y_train, X_val, y_val, X_test, y_test,
    norm_stats: dict, output_dir: Path, metadata: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_mean = norm_stats["X_mean"]
    X_std  = norm_stats["X_std"]
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]

    def nx(X): return (X - X_mean) / X_std
    def ny(y): return (y - y_mean) / y_std

    if HAS_H5PY:
        path = output_dir / "sabr_dataset.h5"
        with h5py.File(path, "w") as f:
            f.attrs["input_cols"] = json.dumps(INPUT_COLS)
            f.attrs["target_col"] = TARGET_COL
            for k, v in metadata.items():
                f.attrs[k] = str(v)
            for name, Xr, yr in [("train", X_train, y_train),
                                   ("val",   X_val,   y_val),
                                   ("test",  X_test,  y_test)]:
                g = f.create_group(name)
                g.create_dataset("X",     data=nx(Xr), compression="gzip")
                g.create_dataset("y",     data=ny(yr), compression="gzip")
                g.create_dataset("X_raw", data=Xr,     compression="gzip")
                g.create_dataset("y_raw", data=yr,     compression="gzip")
            ng = f.create_group("norm")
            ng.create_dataset("X_mean", data=X_mean)
            ng.create_dataset("X_std",  data=X_std)
            ng.attrs["y_mean"] = y_mean
            ng.attrs["y_std"]  = y_std
    else:
        # Fallback: numpy .npz (sin h5py)
        path = output_dir / "sabr_dataset.npz"
        np.savez_compressed(
            path,
            # Datos normalizados
            X_train=nx(X_train), y_train=ny(y_train),
            X_val  =nx(X_val),   y_val  =ny(y_val),
            X_test =nx(X_test),  y_test =ny(y_test),
            # Datos sin normalizar (para análisis)
            X_train_raw=X_train, y_train_raw=y_train,
            X_val_raw  =X_val,   y_val_raw  =y_val,
            X_test_raw =X_test,  y_test_raw =y_test,
            # Estadísticas de normalización
            X_mean=X_mean, X_std=X_std,
            y_mean=np.array([y_mean]), y_std=np.array([y_std]),
        )
        # Guardar metadatos aparte en JSON
        meta_path = output_dir / "dataset_metadata.json"
        with open(meta_path, "w") as mf:
            json.dump({**metadata,
                       "input_cols": INPUT_COLS,
                       "target_col": TARGET_COL,
                       "format": "npz"}, mf, indent=2)

    return path


def load_dataset(
    path: Path | str,
    split: str = "train",
    normalised: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Carga un split del dataset.

    Returns
    -------
    X : np.ndarray, shape (n, 7)
    y : np.ndarray, shape (n,)
    norm_stats : dict con X_mean, X_std, y_mean, y_std
    """
    path = Path(path)

    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            grp    = f[split]
            X_key  = "X"     if normalised else "X_raw"
            y_key  = "y"     if normalised else "y_raw"
            X      = grp[X_key][:]
            y      = grp[y_key][:]
            norm   = f["norm"]
            stats  = {
                "X_mean": norm["X_mean"][:],
                "X_std":  norm["X_std"][:],
                "y_mean": float(norm.attrs["y_mean"]),
                "y_std":  float(norm.attrs["y_std"]),
            }
    else:  # .npz
        data  = np.load(path)
        X_key = f"X_{split}" if normalised else f"X_{split}_raw"
        y_key = f"y_{split}" if normalised else f"y_{split}_raw"
        X     = data[X_key]
        y     = data[y_key]
        stats = {
            "X_mean": data["X_mean"],
            "X_std":  data["X_std"],
            "y_mean": float(data["y_mean"][0]),
            "y_std":  float(data["y_std"][0]),
        }

    return X, y, stats


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def generate_dataset(
    n_samples:  int   = 200_000,
    seed:       int   = 42,
    output_dir: Path  = Path("data"),
    val_frac:   float = 0.10,
    test_frac:  float = 0.10,
    oversample: float = 1.20,
    verbose:    bool  = True,
) -> Path:
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    n_raw = int(n_samples * oversample)

    if verbose:
        print(f"[1/5] Muestreando {n_raw:,} combinaciones de parámetros...")
    params = sample_parameters(n_raw, rng)

    if verbose:
        print(f"[2/5] Evaluando hagan_vol ({n_raw:,} puntos)...")
    t1      = time.perf_counter()
    sigma_B = evaluate_pricer(params)
    if verbose:
        print(f"      {time.perf_counter()-t1:.2f}s  "
              f"({(time.perf_counter()-t1)/n_raw*1e6:.2f} µs/eval)")

    if verbose:
        print(f"[3/5] Filtrando muestras inválidas...")
    params, sigma_B = filter_valid(params, sigma_B)
    n_valid = len(sigma_B)
    if verbose:
        print(f"      Válidas: {n_valid:,} / {n_raw:,} "
              f"({n_valid/n_raw*100:.1f}%)")

    if n_valid > n_samples:
        idx     = rng.choice(n_valid, size=n_samples, replace=False)
        params  = {k: v[idx] for k, v in params.items()}
        sigma_B = sigma_B[idx]
        n_valid = n_samples

    if verbose:
        print(f"[4/5] Split train/val/test...")
    perm    = rng.permutation(n_valid)
    n_test  = int(n_valid * test_frac)
    n_val   = int(n_valid * val_frac)
    n_train = n_valid - n_val - n_test

    X_all = build_input_matrix(params)
    y_all = sigma_B

    X_train, y_train = X_all[perm[:n_train]],            y_all[perm[:n_train]]
    X_val,   y_val   = X_all[perm[n_train:n_train+n_val]], y_all[perm[n_train:n_train+n_val]]
    X_test,  y_test  = X_all[perm[n_train+n_val:]],       y_all[perm[n_train+n_val:]]

    if verbose:
        print(f"      Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    if verbose:
        print(f"[5/5] Normalizando y guardando...")
    norm_stats = compute_normalisation_stats(X_train, y_train)

    if verbose:
        print(f"\n  Estadísticas de normalización (train):")
        print(f"  {'Feature':<12}  {'Media':>10}  {'Std':>10}")
        print(f"  {'-'*36}")
        for i, col in enumerate(INPUT_COLS):
            print(f"  {col:<12}  "
                  f"{norm_stats['X_mean'][i]:>10.5f}  "
                  f"{norm_stats['X_std'][i]:>10.5f}")
        print(f"  {'sigma_B':<12}  "
              f"{norm_stats['y_mean']:>10.5f}  "
              f"{norm_stats['y_std']:>10.5f}")

    metadata = {
        "n_samples": n_valid, "n_train": n_train,
        "n_val": n_val, "n_test": n_test, "seed": seed,
        "pricer": "hagan_vol",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = save_dataset(
        X_train, y_train, X_val, y_val, X_test, y_test,
        norm_stats, Path(output_dir), metadata,
    )

    if verbose:
        import os
        size_mb = os.path.getsize(path) / 1e6
        print(f"\n  Guardado en: {path}  ({size_mb:.1f} MB)")
        print(f"  Tiempo total: {time.perf_counter()-t0:.1f}s")
        print(f"\n  sigma_B — min: {y_all.min():.4f}  max: {y_all.max():.4f}  "
              f"media: {y_all.mean():.4f}  std: {y_all.std():.4f}")

    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int, default=200_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       type=str, default="data")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        print("=" * 55)
        print("Test rápido (10k muestras)")
        print("=" * 55)
        path = generate_dataset(n_samples=10_000, seed=0,
                                output_dir=Path("data/test_run"))

        X_tr, y_tr, stats = load_dataset(path, "train")
        X_va, y_va, _     = load_dataset(path, "val")
        X_te, y_te, _     = load_dataset(path, "test")

        print(f"\n✓ Shapes — Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
        print(f"✓ X_train normalizado — media: {X_tr.mean():.3f}  std: {X_tr.std():.3f}")
        print(f"  (esperado: media≈0, std≈1)")
        print(f"✓ y_train normalizado — media: {y_tr.mean():.3f}  std: {y_tr.std():.3f}")
        assert not np.any(np.isnan(X_tr)), "FALLO: NaN en X_train"
        assert not np.any(np.isnan(y_tr)), "FALLO: NaN en y_train"
        print(f"✓ Sin NaN")
        total = len(y_tr) + len(y_va) + len(y_te)
        print(f"✓ Total: {total:,}  (sin solape entre splits)")
        print("\nTest completado ✓")

    else:
        print("=" * 55)
        print(f"Generando dataset SABR — {args.n:,} muestras")
        print("=" * 55)
        generate_dataset(n_samples=args.n, seed=args.seed,
                         output_dir=Path(args.out))