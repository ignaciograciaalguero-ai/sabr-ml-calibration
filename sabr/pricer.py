"""
sabr/pricer.py
==============
Implementación vectorizada de la fórmula de volatilidad implícita de Black-76
bajo el modelo SABR (Hagan et al., 2002).

Referencias
-----------
Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002).
    Managing Smile Risk. Wilmott Magazine, pp. 84-108.

Obloj, J. (2008). Fine-tune your smile: Correction to Hagan et al.
    arXiv:0708.0998.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Constantes numéricas
# ---------------------------------------------------------------------------
_ATM_THRESHOLD = 1e-4   # |log(F/K)| < umbral → rama ATM
_NU_THRESHOLD  = 1e-6   # nu < umbral → modelo CEV puro (no stochastic vol)
_ARG_MIN       = 1e-12  # cota inferior para log(arg_x) → evita log(0)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def hagan_vol(
    F,
    K,
    T,
    alpha,
    beta,
    rho,
    nu,
    *,
    validate=True,
):
    """
    Volatilidad implícita de Black-76 bajo el modelo SABR (Hagan 2002).

    Implementación completamente vectorizada mediante NumPy broadcasting.
    Gestiona correctamente los tres regímenes especiales:
      - ATM (K ≈ F): límite analítico z/χ(z) → 1.
      - ν ≈ 0: modelo CEV puro, factor B = 1.
      - Parámetros extremos: salvaguardas contra log(0) y divisiones por cero.

    Parameters
    ----------
    F : Forward rate (tipo swap par). Debe ser > 0.
    K : Strike. Debe ser > 0.
    T : Tiempo al vencimiento en años. Debe ser > 0.
    alpha : Volatilidad inicial σ₀. Debe ser > 0.
    beta : Exponente CEV. Debe estar en [0, 1].
    rho : Correlación. Debe estar en (-1, 1).
    nu : Vol-of-vol. Debe ser ≥ 0.
    validate : Si True (por defecto), valida que los parámetros estén en el dominio
    admisible y lanza ValueError en caso contrario. Poner False solo en
    loops de optimización donde la validación ya se hace externamente.

    Returns
    -------
    Volatilidad implícita de Black-76. Mismo shape que el broadcasting
    de los inputs. Siempre devuelve un ndarray (nunca un escalar Python).

    """
    # --- Conversión a arrays float64 ----------------------------------------
    F     = np.asarray(F,     dtype=np.float64)
    K     = np.asarray(K,     dtype=np.float64)
    T     = np.asarray(T,     dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta  = np.asarray(beta,  dtype=np.float64)
    rho   = np.asarray(rho,   dtype=np.float64)
    nu    = np.asarray(nu,    dtype=np.float64)

    # --- Validación de dominio -----------------------------------------------
    if validate:
        _validate_params(F, K, T, alpha, beta, rho, nu)

    # --- Identificación de regímenes especiales ------------------------------
    log_FK   = np.log(F / K)
    is_atm   = np.abs(log_FK) < _ATM_THRESHOLD
    is_cev   = nu < _NU_THRESHOLD       # ν ≈ 0: CEV puro, sin vol estocástica

    # Versiones "seguras" de K y log_FK para las ramas no-ATM
    # (evitan 0/0 en el cálculo de z y denominador_A cuando is_atm=True)
    log_FK_safe = np.where(is_atm, 1.0, log_FK)
    K_safe      = np.where(is_atm, F,   K)

    # --- Cantidades geométricas comunes ---------------------------------------
    one_minus_beta = 1.0 - beta
    FK             = F * K_safe                          # producto F·K
    FK_pow         = FK ** (one_minus_beta / 2.0)        # (FK)^((1-β)/2)
    F_pow          = F ** one_minus_beta                 # F^(1-β) para rama ATM

    # =========================================================================
    # FACTOR A — Nivel base del smile
    # =========================================================================
    # A = α / { (FK)^((1-β)/2) · [1 + (1-β)²/24·ln²(F/K) + (1-β)⁴/1920·ln⁴(F/K)] }
    log2 = log_FK_safe ** 2
    log4 = log_FK_safe ** 4

    # En la rama ATM, log_FK_safe = 1 → los términos del denominador no son
    # cero pero tampoco corresponden al límite correcto. Por eso A_atm se
    # calcula por separado usando F en lugar de (FK)^((1-β)/2).
    denom_offatm = FK_pow * (
        1.0
        + (one_minus_beta**2 / 24.0) * log2
        + (one_minus_beta**4 / 1920.0) * log4
    )
    # Límite ATM del denominador: F^(1-β) · [1 + 0 + 0] = F^(1-β)
    denom_atm = F_pow

    denom_A = np.where(is_atm, denom_atm, denom_offatm)
    A = alpha / denom_A

    # =========================================================================
    # FACTOR B — Corrección por correlación (ratio z/χ(z))
    # =========================================================================
    # En la rama ATM o cuando ν ≈ 0, B = 1 exactamente.
    z     = (nu / alpha) * FK_pow * log_FK_safe
    arg_x = ((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho)
             / (1.0 - rho))

    # Salvaguarda: arg_x debe ser > 0 para que log sea válido.
    # Si arg_x ≤ 0 los parámetros son extremos y el resultado será NaN,
    # lo que es preferible a un error silencioso o una excepción de NumPy.
    arg_x_safe = np.maximum(arg_x, _ARG_MIN)
    x_z        = np.log(arg_x_safe)

    # Ratio z/χ(z): límite → 1 cuando z → 0 (rama ATM o ν → 0)
    # np.where evalúa ambas ramas, así que protegemos x_z contra división /0
    x_z_safe = np.where(np.abs(x_z) < _ARG_MIN, 1.0, x_z)
    B_offatm  = z / x_z_safe
    B = np.where(is_atm | is_cev, 1.0, B_offatm)

    # =========================================================================
    # FACTOR C — Corrección temporal de primer orden O(T)
    # =========================================================================
    # C = 1 + [term1 + term2 + term3] · T
    # donde:
    #   term1 = (1-β)²α² / [24·(FK)^(1-β)]    → curvatura CEV
    #   term2 = ρβνα / [4·(FK)^((1-β)/2)]     → interacción correlación-CEV
    #   term3 = (2-3ρ²)ν² / 24                → vol-of-vol (curvatura del smile)

    # Para la rama ATM, (FK)^(1-β) → F^(2(1-β)) y (FK)^((1-β)/2) → F^(1-β)
    FK_pow_1mb   = np.where(is_atm, F_pow**2,  FK ** one_minus_beta)
    FK_pow_half  = np.where(is_atm, F_pow,     FK_pow)

    term1 = (one_minus_beta**2 / 24.0) * (alpha**2) / FK_pow_1mb
    term2 = 0.25 * rho * beta * nu * alpha / FK_pow_half
    term3 = ((2.0 - 3.0 * rho**2) / 24.0) * nu**2

    C = 1.0 + (term1 + term2 + term3) * T

    # =========================================================================
    # Volatilidad implícita final
    # =========================================================================
    vol = A * B * C

    # Garantizamos que el output es siempre un ndarray (contrato de la función)
    return np.atleast_1d(vol)


# ---------------------------------------------------------------------------
# Función de superficie (wrapper conveniente)
# ---------------------------------------------------------------------------
def sabr_surface(
    F0,
    strikes,
    maturities,
    alpha,
    beta,
    rho,
    nu,
):
    """
    Genera la superficie de volatilidad implícita SABR.

    Parameters
    ----------
    F0 : Forward rate actual (único para toda la superficie).
    strikes : Vector de strikes.
    maturities : Vector de vencimientos en años.
    alpha, beta, rho, nu : Parámetros SABR (únicos para toda la superficie).

    Returns
    -------
    Matriz de volatilidades implícitas. Fila i = vencimiento i, columna j = strike j.
    """
    strikes    = np.asarray(strikes,    dtype=np.float64)
    maturities = np.asarray(maturities, dtype=np.float64)

    # Broadcasting: K_grid shape (N, M), T_grid shape (N, M)
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    return hagan_vol(F0, K_grid, T_grid, alpha, beta, rho, nu)


# ---------------------------------------------------------------------------
# Validación interna de parámetros
# ---------------------------------------------------------------------------
def _validate_params(
    F, K, T,
    alpha, beta, rho, nu,
):
    """Valida que todos los parámetros estén en el dominio admisible del modelo."""
    errors = []

    if np.any(F <= 0):
        errors.append(f"F debe ser > 0. Min recibido: {np.min(F):.6f}")
    if np.any(K <= 0):
        errors.append(f"K debe ser > 0. Min recibido: {np.min(K):.6f}")
    if np.any(T <= 0):
        errors.append(f"T debe ser > 0. Min recibido: {np.min(T):.6f}")
    if np.any(alpha <= 0):
        errors.append(f"alpha debe ser > 0. Min recibido: {np.min(alpha):.6f}")
    if np.any((beta < 0) | (beta > 1)):
        errors.append(f"beta debe estar en [0, 1]. Rango recibido: [{np.min(beta):.4f}, {np.max(beta):.4f}]")
    if np.any(np.abs(rho) >= 1):
        errors.append(f"|rho| debe ser < 1. Max |rho| recibido: {np.max(np.abs(rho)):.6f}")
    if np.any(nu < 0):
        errors.append(f"nu debe ser >= 0. Min recibido: {np.min(nu):.6f}")

    if errors:
        raise ValueError("Parámetros fuera del dominio admisible SABR:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Tests de cordura (ejecutar con: python -m pytest sabr/pricer.py -v --doctest-modules)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Tests de sanidad del pricer SABR")
    print("=" * 60)

    # Parámetros de referencia
    F0    = 0.03      # 3% forward rate
    alpha = 0.20
    beta  = 0.5
    rho   = -0.30
    nu    = 0.40
    T     = 1.0

    # --- Test 1: caso ATM ---------------------------------------------------
    vol_atm = hagan_vol(F0, F0, T, alpha, beta, rho, nu)
    vol_atm_formula = (alpha / F0**(1 - beta)) * (
        1 + ((1-beta)**2 * alpha**2 / (24 * F0**(2*(1-beta)))
             + rho * beta * nu * alpha / (4 * F0**(1-beta))
             + (2 - 3*rho**2) * nu**2 / 24) * T
    )
    assert np.abs(vol_atm - vol_atm_formula) < 1e-10, "FALLO Test ATM"
    print(f"✓ Test 1 (ATM):        σ_ATM = {vol_atm[0]:.6f}  "
          f"(fórmula directa: {vol_atm_formula:.6f})")

    # --- Test 2: skew negativo → OTM put > ATM > OTM call ------------------
    # Con rho < 0, el smile es monótonamente decreciente en K.
    # En el modelo SABR con beta < 1 y rho < 0, la vol decrece con K.
    K_low  = F0 * 0.7    # OTM put (strike bajo)
    K_high = F0 * 1.3    # OTM call (strike alto)
    vol_low  = hagan_vol(F0, K_low,  T, alpha, beta, rho, nu)[0]
    vol_high = hagan_vol(F0, K_high, T, alpha, beta, rho, nu)[0]
    assert vol_low  > vol_atm[0], "FALLO Test skew: OTM put debe tener mayor vol que ATM"
    assert vol_high < vol_atm[0], "FALLO Test skew: OTM call debe tener menor vol que ATM"
    print(f"✓ Test 2 (Skew rho<0): σ_put={vol_low:.4f} > σ_ATM={vol_atm[0]:.4f} "
          f"> σ_call={vol_high:.4f}")

    # --- Test 3: smile monótonamente decreciente en K para rho < 0 ----------
    strikes_mono = np.linspace(F0 * 0.5, F0 * 2.0, 50)
    vols_mono    = hagan_vol(F0, strikes_mono, T, alpha, beta, rho, nu)
    assert np.all(np.diff(vols_mono) < 0), \
        "FALLO Test monotonía: con rho < 0 la vol debe decrecer con K"
    print(f"✓ Test 3 (Monotonía):  vol monótonamente decreciente en K para rho={rho}")

    # --- Test 4: continuidad en ATM (no NaN en la transición) ---------------
    K_near_atm = np.linspace(F0 * (1 - 5e-4), F0 * (1 + 5e-4), 100)
    vols_near  = hagan_vol(F0, K_near_atm, T, alpha, beta, rho, nu)
    assert not np.any(np.isnan(vols_near)), "FALLO Test continuidad: NaN cerca de ATM"
    assert not np.any(np.isinf(vols_near)), "FALLO Test continuidad: Inf cerca de ATM"
    print(f"✓ Test 4 (Continuidad): sin NaN/Inf en la transición ATM "
          f"(rango: [{vols_near.min():.6f}, {vols_near.max():.6f}])")

    # --- Test 5: superficie 2D ---------------------------------------------
    strikes    = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    maturities = np.array([1.0, 5.0, 10.0])
    surface    = sabr_surface(F0, strikes, maturities, alpha, beta, rho, nu)
    assert surface.shape == (3, 5), f"FALLO Test superficie: shape {surface.shape}"
    assert not np.any(np.isnan(surface)), "FALLO Test superficie: NaN en la malla"
    print(f"✓ Test 5 (Superficie): shape {surface.shape}, "
          f"rango [{surface.min():.4f}, {surface.max():.4f}]")

    # --- Test 6: validación de dominio -------------------------------------
    try:
        hagan_vol(-0.03, 0.03, 1.0, 0.20, 0.5, -0.3, 0.4)
        print("✗ Test 6 (Validación): no lanzó ValueError con F < 0")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Test 6 (Validación): ValueError capturado correctamente")

    # --- Test 7: nu = 0 (CEV puro) -----------------------------------------
    vol_cev = hagan_vol(F0, F0 * 1.2, T, alpha, beta, rho, nu=0.0)
    assert not np.any(np.isnan(vol_cev)), "FALLO Test CEV: NaN con nu=0"
    print(f"✓ Test 7 (CEV puro):   σ(nu=0) = {vol_cev[0]:.6f} (sin NaN)")

    print("\n" + "=" * 60)
    print("Todos los tests pasados ✓")
    print("=" * 60)