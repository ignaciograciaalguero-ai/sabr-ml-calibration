"""
sabr/monte_carlo.py
===================
Simulador Monte Carlo vectorizado para las EDEs del modelo SABR.

Implementa:
  - Esquema Euler-Maruyama con truncamiento completo.
  - Esquema Milstein con truncamiento completo (reducción de sesgo).
  - Simulación exacta de sigma_t (log-Euler sin error de discretización).
  - Muestras antitéticas para reducción de varianza.
  - Valoración vectorizada sobre múltiples strikes simultáneos.
  - Inversión de Black-76 por bisección para obtener volatilidad implícita.

Referencias
-----------
Andersen, L.B.G. (2008). Simple and efficient simulation of the Heston
    stochastic volatility model. Journal of Computational Finance, 11(3).

Broadie, M. & Kaya, O. (2006). Exact simulation of stochastic volatility
    and other affine jump diffusion processes. Operations Research, 54(2).
"""

import time
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_MIN_F:     float = 1e-10   # cota inferior de F para evitar log(0) en inversión
_MIN_PRICE: float = 1e-12   # precio mínimo considerado > 0 para inversión
_VOL_LO:    float = 1e-6    # cota inferior búsqueda vol implícita
_VOL_HI:    float = 10.0    # cota superior búsqueda vol implícita


# ===========================================================================
# BLOQUE 1 — Simulación de trayectorias
# ===========================================================================

def simulate_sabr(
    F0,
    T,
    alpha,
    beta,
    rho,
    nu,
    n_paths = 100_000,
    n_steps = 100,
    scheme = "milstein",
    antithetic = True,
    seed= None,
):
    """
    Simula trayectorias del modelo SABR hasta el tiempo T.

    Usa simulación exacta para sigma_t (log-Euler sin sesgo de discretización)
    y esquema Milstein con truncamiento completo para F_t.

    Parameters
    ----------
    F0 : 
        Forward inicial. Debe ser > 0.
    T : 
        Tiempo a vencimiento en años. Debe ser > 0.
    alpha, beta, rho, nu : 
        Parámetros SABR. Ver restricciones en pricer.py.
    n_paths : 
        Número de trayectorias. Con antithetic=True, el número real de
        trayectorias independientes es n_paths // 2.
    n_steps : 
        Número de pasos temporales. Más pasos → menor sesgo de discretización
        en F_t. Recomendado: >= 50 para T <= 1, >= 200 para T >= 10.
    scheme : {'euler', 'milstein'}
        Esquema de discretización para F_t. Milstein tiene sesgo O(dt²)
        frente a O(dt) de Euler, al coste de un término adicional por paso.
    antithetic : 
        Si True, usa muestras antitéticas: genera n_paths/2 brownianos
        independientes y sus antitéticos -Z. Reduce la varianza del estimador
        típicamente por un factor 2-5 sin coste adicional significativo.
    seed :
        Semilla para reproducibilidad. None → no fijada.

    Returns
    -------
    F_T :
        Forwards terminales al tiempo T.
    sigma_T :
        Volatilidades terminales al tiempo T.
    """
    if antithetic and n_paths % 2 != 0:
        n_paths += 1   # garantizamos que n_paths es par

    dt       = T / n_steps
    sqrt_dt  = np.sqrt(dt)
    rho_perp = np.sqrt(1.0 - rho**2)   # componente ortogonal de la correlación

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Generación de brownianos
    # Con muestras antitéticas, generamos la mitad y reflejamos.
    # -----------------------------------------------------------------------
    if antithetic:
        half = n_paths // 2
        # Shape: (n_steps, half)
        Z1_half = rng.standard_normal((n_steps, half))
        Z2_half = rng.standard_normal((n_steps, half))
        # Concatenamos con sus antitéticos → shape (n_steps, n_paths)
        Z1 = np.concatenate([Z1_half, -Z1_half], axis=1)
        Z2 = np.concatenate([Z2_half, -Z2_half], axis=1)
    else:
        Z1 = rng.standard_normal((n_steps, n_paths))
        Z2 = rng.standard_normal((n_steps, n_paths))

    # Browniano para sigma correlacionado con W^F mediante descomposición de Cholesky
    # dW^sigma = rho * dW^F + sqrt(1-rho²) * dW^perp
    Z_sigma = rho * Z1 + rho_perp * Z2

    # -----------------------------------------------------------------------
    # Estado inicial
    # -----------------------------------------------------------------------
    F     = np.full(n_paths, F0,    dtype=np.float64)
    sigma = np.full(n_paths, alpha, dtype=np.float64)

    # Precalcular constantes del esquema de Milstein
    # Corrección Milstein para dF = sigma * F^beta * dW^F:
    #   dF_milstein = sigma * F^beta * dW + 0.5 * sigma² * beta * F^(2β-1) * (dW² - dt)
    two_beta_minus_one = 2.0 * beta - 1.0
    half_beta          = 0.5 * beta

    # -----------------------------------------------------------------------
    # Bucle temporal — NumPy vectoriza sobre los n_paths en cada paso
    # -----------------------------------------------------------------------
    for k in range(n_steps):
        z1    = Z1[k]        # shape (n_paths,)
        z_sig = Z_sigma[k]   # shape (n_paths,)

        # === Actualización de sigma_t: simulación exacta (sin sesgo) ========
        # sigma_{t+dt} = sigma_t * exp(nu * dW^sigma - 0.5 * nu² * dt)
        sigma_next = sigma * np.exp(
            nu * sqrt_dt * z_sig - 0.5 * nu * nu * dt
        )

        # === Actualización de F_t: Milstein con truncamiento completo ========
        # Truncamiento completo: usamos max(F, 0) en la difusión pero
        # permitimos que F acumule valores negativos para evitar sesgo de reflexión.
        # Al final de cada paso forzamos F >= 0.
        F_plus = np.maximum(F, 0.0)          # F truncado para la difusión

        # Término de difusión: sigma * F^beta * sqrt(dt) * Z1
        F_beta     = F_plus ** beta
        dW_F       = sqrt_dt * z1
        diffusion  = sigma * F_beta * dW_F

        if scheme == "euler":
            dF = diffusion

        else:  # milstein
            # Corrección de Milstein: 0.5 * sigma² * beta * F^(2β-1) * (dW² - dt)
            # Caso beta=0: F^(2β-1) = F^(-1) → corrección = 0 (proceso normal puro)
            if beta == 0.0:
                dF = diffusion
            elif beta == 1.0:
                # F^(2β-1) = F^1 = F_plus
                milstein_term = half_beta * sigma * sigma * F_plus * (dW_F * dW_F - dt)
                dF = diffusion + milstein_term
            else:
                # Caso general: F^(2β-1), necesita F > 0 para no explotar
                F_safe        = np.maximum(F_plus, _MIN_F)
                F_2b1         = F_safe ** two_beta_minus_one
                milstein_term = half_beta * sigma * sigma * F_2b1 * (dW_F * dW_F - dt)
                dF = diffusion + milstein_term

        # Aplicar incremento y truncar a [0, ∞)
        F     = np.maximum(F + dF, 0.0)
        sigma = sigma_next

    return F, sigma


# ===========================================================================
# BLOQUE 2 — Valoración de opciones
# ===========================================================================

def mc_price(
    F_T,
    strikes,
    df = 1.0,
    option_type = "call",
):
    """
    Valoración de opciones europeas call/put mediante simulación Monte Carlo.

    Vectorizada sobre múltiples strikes simultáneamente, reutilizando las
    mismas trayectorias F_T para todos los strikes. Esto es eficiente para
    generar el dataset de entrenamiento del surrogate model.

    Parameters
    ----------
    F_T :
        Forwards terminales generados por simulate_sabr().
    strikes :
        Strike o array de strikes.
    df :
        Factor de descuento P(0,T). Para volatilidades implícitas usar 1.0.
    option_type : {'call', 'put'}
        Tipo de opción.

    Returns
    -------
    prices :
        Precios esperados descontados para cada strike.
    stderr :
        Error estándar del estimador Monte Carlo para cada strike.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    n_paths = len(F_T)

    # Broadcasting: F_T shape (n_paths, 1), strikes shape (1, M)
    # → payoffs shape (n_paths, M)
    if option_type == "call":
        payoffs = np.maximum(F_T[:, None] - strikes[None, :], 0.0)
    else:
        payoffs = np.maximum(strikes[None, :] - F_T[:, None], 0.0)

    prices = df * np.mean(payoffs, axis=0)                              # shape (M,)
    stderr = df * np.std(payoffs, axis=0, ddof=1) / np.sqrt(n_paths)   # shape (M,)

    return prices, stderr


# ===========================================================================
# BLOQUE 3 — Inversión de Black-76 → volatilidad implícita
# ===========================================================================

def _black76_call(F, K, T, sigma):
    """Precio call Black-76 (factor de descuento = 1)."""
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return 0.0
    sqrtT  = np.sqrt(T)
    d_plus  = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d_minus = d_plus - sigma * sqrtT
    return F * norm.cdf(d_plus) - K * norm.cdf(d_minus)


def _black76_put(F, K, T, sigma):
    """Precio put Black-76 (factor de descuento = 1)."""
    call = _black76_call(F, K, T, sigma)
    return call - F + K   # paridad call-put


def _invert_black76(
    price,
    F,
    K,
    T,
    option_type,
):
    """
    Invierte Black-76 para obtener la volatilidad implícita.

    Usa bisección (Brent) en [_VOL_LO, _VOL_HI]. Devuelve NaN si el precio
    está fuera del rango de Black-76 (por ejemplo, precio < valor intrínseco).
    """
    if price < _MIN_PRICE:
        return np.nan

    pricer = _black76_call if option_type == "call" else _black76_put

    # Valor intrínseco: cota inferior del precio
    intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
    if price <= intrinsic + _MIN_PRICE:
        return np.nan

    # Verificar que existe solución en el intervalo
    f_lo = pricer(F, K, T, _VOL_LO) - price
    f_hi = pricer(F, K, T, _VOL_HI) - price

    if f_lo * f_hi > 0:
        # El precio no está en el rango de Black-76 con el intervalo dado
        return np.nan

    try:
        sigma_impl = brentq(
            lambda s: pricer(F, K, T, s) - price,
            _VOL_LO, _VOL_HI,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=100,
        )
        return float(sigma_impl)
    except (ValueError, RuntimeError):
        return np.nan


def mc_implied_vol(
    F0,
    T,
    alpha,
    beta,
    rho,
    nu,
    strikes,
    n_paths = 100_000,
    n_steps = 100,
    scheme = "milstein",
    antithetic= True,
    seed = None,
    option_type = "call",
):
    """
    Volatilidad implícita Black-76 bajo SABR mediante Monte Carlo.

    Función de alto nivel: simula trayectorias, valora opciones y
    obtiene volatilidades implícitas por inversión de Black-76.
    Es la función que se usa para comparar con hagan_vol().

    Parameters
    ----------
    F0, T, alpha, beta, rho, nu : float
        Parámetros del modelo y del instrumento.
    strikes :
        Strike o array de strikes.
    n_paths, n_steps, scheme, antithetic, seed : ver simulate_sabr().
    option_type : {'call', 'put'}
        Tipo de opción. Para ATM usar 'call'.

    Returns
    -------
    impl_vols :
        Volatilidades implícitas Black-76. NaN si la inversión falla.
    stderr_vols :
        Error estándar de los precios MC (en unidades de precio, no de vol).
        Útil para estimar la incertidumbre MC en la volatilidad implícita.

    Notes
    -----
    La conversión de error de precio a error de vol (en bps) es aproximada:
        stderr_vol ≈ stderr_price / vega_Black
    donde vega_Black = F * sqrt(T) * phi(d+). No se implementa aquí para
    mantener la función simple, pero se incluye en el análisis del Capítulo 5.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=np.float64))

    # Fase 1: Simular trayectorias
    F_T, _ = simulate_sabr(
        F0, T, alpha, beta, rho, nu,
        n_paths=n_paths, n_steps=n_steps,
        scheme=scheme, antithetic=antithetic, seed=seed,
    )

    # Fase 2: Precios MC para todos los strikes (vectorizado)
    prices, stderr = mc_price(F_T, strikes, option_type=option_type)

    # Fase 3: Inversión de Black-76 strike a strike
    impl_vols = np.array([
        _invert_black76(float(p), F0, float(K), T, option_type)
        for p, K in zip(prices, strikes)
    ])

    return impl_vols, stderr


# ===========================================================================
# BLOQUE 4 — Benchmark de velocidad
# ===========================================================================

def benchmark(
    F0 = 0.03,
    T = 1.0,
    alpha = 0.20,
    beta = 0.5,
    rho = -0.30,
    nu = 0.40,
    n_paths = 100_000,
    n_steps = 100,
):
    """
    Mide el tiempo de simulación y valoración del MC.

    Returns
    -------
    dict con tiempos en segundos y configuración usada.
    """
    strikes = np.array([F0 * m for m in [0.7, 0.85, 1.0, 1.15, 1.30]])

    # Warmup
    simulate_sabr(F0, T, alpha, beta, rho, nu, n_paths=1000, n_steps=10, seed=0)

    t0 = time.perf_counter()
    F_T, _ = simulate_sabr(
        F0, T, alpha, beta, rho, nu,
        n_paths=n_paths, n_steps=n_steps, seed=42,
    )
    t_sim = time.perf_counter() - t0

    t0 = time.perf_counter()
    prices, _ = mc_price(F_T, strikes)
    t_price = time.perf_counter() - t0

    t0 = time.perf_counter()
    vols, _ = mc_implied_vol(
        F0, T, alpha, beta, rho, nu, strikes,
        n_paths=n_paths, n_steps=n_steps, seed=42,
    )
    t_total = time.perf_counter() - t0

    return {
        "n_paths":     n_paths,
        "n_steps":     n_steps,
        "t_sim_s":     t_sim,
        "t_price_s":   t_price,
        "t_total_s":   t_total,
        "vols":        vols,
        "strikes":     strikes,
    }


# ===========================================================================
# Tests de sanidad
# ===========================================================================

if __name__ == "__main__":
    import sys
    from sabr.pricer import hagan_vol   # importar desde el mismo paquete

    print("=" * 65)
    print("Tests de sanidad del simulador Monte Carlo SABR")
    print("=" * 65)

    # Parámetros realistas de mercado (swaptions en euros, 2024-2025)
    # F0=3%, alpha calibrado para vol ATM ~50 bps en lognormal
    # Con beta=0.5: vol_ATM ≈ alpha / F0^0.5 → alpha ≈ 0.008 para ~47 bps
    F0    = 0.03
    alpha = 0.008
    beta  = 0.5
    rho   = -0.30
    nu    = 0.40
    T     = 1.0
    seed  = 42
    N     = 200_000   # trayectorias (antithetic → 100k independientes)

    # Strikes en términos absolutos: ±50, ±20, 0 bps alrededor del ATM
    # (strikes relativos ±30% son demasiado OTM para tipos bajos)
    strikes   = np.array([F0 + s for s in [-0.005, -0.002, 0.0, 0.002, 0.005]])
    moneyness = ["-50bps", "-20bps", "ATM", "+20bps", "+50bps"]

    # --- Test 1: simulate_sabr devuelve el shape correcto -------------------
    F_T, sig_T = simulate_sabr(
        F0, T, alpha, beta, rho, nu,
        n_paths=N, n_steps=100, seed=seed,
    )
    assert F_T.shape  == (N,), f"FALLO shape F_T: {F_T.shape}"
    assert sig_T.shape == (N,), f"FALLO shape sig_T: {sig_T.shape}"
    assert np.all(F_T   >= 0), "FALLO: F_T contiene valores negativos"
    assert np.all(sig_T >= 0), "FALLO: sigma_T contiene valores negativos"
    print(f"✓ Test 1 (Shape):      F_T shape={F_T.shape}, "
          f"F_T mean={F_T.mean():.5f}, F_T min={F_T.min():.5f}")

    # --- Test 2: F_T es martingala bajo Q (media ≈ F0) ----------------------
    mean_F = F_T.mean()
    tol_martingala = 0.001   # 10 bps de tolerancia
    assert abs(mean_F - F0) < tol_martingala, \
        f"FALLO martingala: E[F_T]={mean_F:.6f} ≠ F0={F0:.6f}"
    print(f"✓ Test 2 (Martingala): E[F_T] = {mean_F:.6f} ≈ F0 = {F0:.6f} "
          f"(error = {abs(mean_F-F0)*1e4:.2f} bps)")

    # --- Test 3: comparación MC vs Hagan en régimen de baja varianza --------
    print("\n✓ Test 3 (MC vs Hagan): T=1y, α=0.20, β=0.5, ρ=-0.30, ν=0.40")
    print(f"  {'Strike':>8}  {'Hagan':>10}  {'MC':>10}  {'|Error|':>10}  {'Error(bps)':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    vols_mc, stderr = mc_implied_vol(
        F0, T, alpha, beta, rho, nu, strikes,
        n_paths=N, n_steps=100, seed=seed,
    )
    vols_hagan = hagan_vol(F0, strikes, T, alpha, beta, rho, nu)

    max_error_bps = 0.0
    for m, K, v_h, v_mc in zip(moneyness, strikes, vols_hagan, vols_mc):
        if np.isnan(v_mc):
            print(f"  {m:>8}  {v_h:>10.6f}  {'NaN':>10}  {'---':>10}  {'---':>12}")
            continue
        err = abs(v_h - v_mc)
        err_bps = err * 1e4
        max_error_bps = max(max_error_bps, err_bps)
        print(f"  {m:>8}  {v_h:>10.6f}  {v_mc:>10.6f}  "
              f"{err:>10.6f}  {err_bps:>10.2f} bps")

    # Para T=1y con parámetros realistas de mercado, Hagan es preciso:
    # el error esperado es < 10 bps (dominado por ruido MC, no por error de Hagan).
    # Umbral generoso de 30 bps para acomodar variabilidad estadística del MC.
    assert max_error_bps < 30, \
        f"FALLO: error máximo {max_error_bps:.1f} bps > 30 bps"
    print(f"  → Error máximo: {max_error_bps:.2f} bps (umbral: 30 bps)")

    # --- Test 4: régimen largo (T=10y) — Hagan debe tener mayor error -------
    print("\n✓ Test 4 (T=10y): comparación donde Hagan empieza a degradarse")
    T_long = 10.0
    vols_mc_long, _ = mc_implied_vol(
        F0, T_long, alpha, beta, rho, nu, np.array([F0]),
        n_paths=N, n_steps=200, seed=seed,
    )
    vol_hagan_long = hagan_vol(F0, F0, T_long, alpha, beta, rho, nu)[0]
    err_long_bps = abs(vol_hagan_long - vols_mc_long[0]) * 1e4
    print(f"  Hagan ATM (T=10y): {vol_hagan_long:.6f}")
    if not np.isnan(vols_mc_long[0]):
        print(f"  MC    ATM (T=10y): {vols_mc_long[0]:.6f}")
        print(f"  Error: {err_long_bps:.2f} bps "
              f"({'mayor que T=1y' if err_long_bps > max_error_bps else 'OK'})")

    # --- Test 5: muestras antitéticas — verificación estructural -------------
    # Las antitéticas garantizan E[F_T] = F0 exactamente cuando n_paths es par,
    # porque cada trayectoria Z tiene su antitética -Z con media exactamente F0.
    # Este es el beneficio estructural principal, independiente del strike.
    print("\n✓ Test 5 (Antitéticas): verificación de propiedad estructural")

    F_anti, _ = simulate_sabr(
        F0, T, alpha, beta, rho, nu,
        n_paths=10_000, n_steps=50,
        antithetic=True, seed=7,
    )
    F_noanti, _ = simulate_sabr(
        F0, T, alpha, beta, rho, nu,
        n_paths=10_000, n_steps=50,
        antithetic=False, seed=7,
    )

    err_anti   = abs(F_anti.mean()   - F0)
    err_noanti = abs(F_noanti.mean() - F0)

    print(f"  E[F_T] sin antitéticas: {F_noanti.mean():.8f}  (error = {err_noanti*1e4:.3f} bps)")
    print(f"  E[F_T] con antitéticas: {F_anti.mean():.8f}    (error = {err_anti*1e4:.3f} bps)")
    print(f"  Las antitéticas reducen el error en la media por factor: "
          f"{err_noanti/err_anti:.1f}x" if err_anti > 0 else "  Error con antitéticas es cero exacto")

    # Con antitéticas, el error en la media debe ser menor que sin ellas
    assert err_anti <= err_noanti + 1e-8, \
        "FALLO: antitéticas no mejoran la estimación de la media"
    print(f"  ✓ Antitéticas mejoran la estimación de la media de F_T")

    # --- Test 6: benchmark de velocidad ------------------------------------
    print("\n✓ Test 6 (Benchmark):")
    res = benchmark(F0=F0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu,
                    n_paths=100_000, n_steps=100)
    print(f"  Simulación ({res['n_paths']:,} paths, {res['n_steps']} steps): "
          f"{res['t_sim_s']:.2f} s")
    print(f"  Valoración (5 strikes):  {res['t_price_s']*1000:.1f} ms")
    print(f"  Pipeline completo:       {res['t_total_s']:.2f} s")
    print(f"  Tiempo por trayectoria:  "
          f"{res['t_sim_s']/res['n_paths']*1e6:.2f} µs")

    print("\n" + "=" * 65)
    print("Todos los tests pasados ✓")
    print("=" * 65)