"""
experiments/01_pricer_benchmark.py
===================================
Script de benchmark de rendimiento y visualización 3D/2D para el pricer SABR.

Funcionalidades:
  1. Medición de latencia y throughput (evaluaciones/segundo) para grids de distinto tamaño.
  2. Generación de gráfico 3D de la superficie de volatilidad implícita (K, T -> sigma).
  3. Generación de gráfico 2D comparativo del smile/skew de volatilidad para diferentes vencimientos.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar las funciones del pricer
from sabr.pricer import hagan_vol, sabr_surface

# Configuración estética de Matplotlib
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
})


def run_performance_benchmark():
    """Mide el tiempo de ejecución de la fórmula de Hagan para diferentes tamaños de lote."""
    print("=" * 65)
    print(" BENCHMARK DE RENDIMIENTO (NUMPY VECTORIZADO)")
    print("=" * 65)

    F0 = 0.03
    alpha = 0.02
    beta = 0.5
    rho = -0.3
    nu = 0.4

    # Diferentes escalas de evaluación
    grid_sizes = [(10, 10), (50, 50), (100, 100), (500, 500), (1000, 1000)]

    print(f"{'Vencimientos (N)':<16} | {'Strikes (M)':<14} | {'Puntos Totales':<16} | {'Tiempo (ms)':<14} | {'Puntos / sec':<16}")
    print("-" * 85)

    for n_t, n_k in grid_sizes:
        strikes = np.linspace(0.005, 0.08, n_k)
        maturities = np.linspace(0.1, 10.0, n_t)

        # Warm-up run
        _ = sabr_surface(F0, strikes, maturities, alpha, beta, rho, nu)

        # Medición de tiempo con perf_counter
        n_iters = 20
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = sabr_surface(F0, strikes, maturities, alpha, beta, rho, nu)
        elapsed_avg = (time.perf_counter() - start) / n_iters

        total_points = n_t * n_k
        throughput = total_points / elapsed_avg

        print(f"{n_t:<16} | {n_k:<14} | {total_points:<16,d} | {elapsed_avg * 1000:<14.3f} | {throughput:<16,.0f}")

    print("-" * 85 + "\n")


def plot_volatility_surface():
    """Genera y guarda las visualizaciones 3D y 2D de la superficie de volatilidad SABR."""
    F0 = 0.03
    alpha = 0.02
    beta = 0.5
    rho = -0.35
    nu = 0.45

    strikes = np.linspace(0.01, 0.06, 80)
    maturities = np.linspace(0.25, 5.0, 50)

    # Calcular superficie 2D (Maturities x Strikes)
    surface = sabr_surface(F0, strikes, maturities, alpha, beta, rho, nu)
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(14, 6))

    # -------------------------------------------------------------------------
    # Gráfico 1: Superficie 3D (K, T -> Vol)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(
        K_grid * 100,
        T_grid,
        surface * 100,
        cmap="viridis",
        edgecolor="none",
        alpha=0.85,
    )
    ax1.set_xlabel("Strike K (%)", labelpad=10)
    ax1.set_ylabel("Vencimiento T (Años)", labelpad=10)
    ax1.set_zlabel("Vol. Implícita Black (%)", labelpad=10)
    ax1.set_title(r"Superficie SABR 3D ($\alpha=0.02, \beta=0.5, \rho=-0.35, \nu=0.45$)", pad=15)
    ax1.view_init(elev=25, azim=-125)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # -------------------------------------------------------------------------
    # Gráfico 2: Cortes 2D de Volatility Smile a distintos vencimientos
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    selected_maturities = [0.25, 0.5, 1.0, 2.0, 5.0]

    for T_val in selected_maturities:
        vols = hagan_vol(F0, strikes, T_val, alpha, beta, rho, nu, validate=True)
        ax2.plot(strikes * 100, vols * 100, label=f"T = {T_val}y", linewidth=1.8)

    ax2.axvline(F0 * 100, color="gray", linestyle="--", alpha=0.7, label=f"ATM (F0 = {F0*100:.1f}%)")
    ax2.set_xlabel("Strike K (%)")
    ax2.set_ylabel("Volatilidad Implícita (%)")
    ax2.set_title("Evolución del Skew/Smile según el Vencimiento (T)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()

    # Guardar figura en el directorio de experimentos
    output_dir = Path(__file__).parent
    output_path = output_dir / "sabr_surface_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Gráfico guardado correctamente en: {output_path}")
    plt.show()


if __name__ == "__main__":
    run_performance_benchmark()
    plot_volatility_surface()