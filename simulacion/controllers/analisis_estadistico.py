"""
Análisis estadístico comparativo — STHWP vs SUBWP (E2.1)
=========================================================
Ejecutar desde simulacion/controllers/:
    python3 analisis_estadistico.py

Produce:
  1. Tabla comparativa STHWP vs SUBWP por goal con IC 95% (Wilson) y p-valor (chi²)
  2. Impacto del peatón por sistema (E2.1 con peatón vs E2.1 sin peatón)
  3. Tabla resumen global por experimento con IC bootstrap
  4. Goals con diferencia estadísticamente significativa (p < 0.05)
"""

import csv
import math
import os
import sys
from scipy import stats as scipy_stats
import numpy as np

BASE_STHWP = "rl_train_STHWP/experimentos/resultados"
BASE_SUBWP = "rl_train_SUB_WP_continuo/experimentos/resultados"
SEEDS      = [42, 123, 524]
N_EP       = 100  # episodios por goal por seed


# ── Utilidades ──────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """IC de Wilson al 95% para una proporción k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def leer_exitos_por_goal(paths):
    """Lee varios CSVs y devuelve {goal_id: exitos_totales, n_total}."""
    conteo = {}
    n_total = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                gid = row["goal_id"]
                conteo[gid] = conteo.get(gid, 0) + (1 if row["resultado"] == "exito" else 0)
                n_total[gid] = n_total.get(gid, 0) + 1
    return conteo, n_total


def agregar_seeds(prefix, seeds, base_dir):
    paths = [os.path.join(base_dir, f"{prefix}_s{s}.csv") for s in seeds]
    return leer_exitos_por_goal(paths)


def tasa_global(exitos, totales):
    ex = sum(exitos.values())
    to = sum(totales.values())
    return ex / to * 100 if to > 0 else 0.0, ex, to


# ── Cargar datos ─────────────────────────────────────────────────────────────

sthwp_e21_ex,  sthwp_e21_n  = agregar_seeds("sthwp_infer_e2_1",       SEEDS, BASE_STHWP)
subwp_e21_ex,  subwp_e21_n  = agregar_seeds("subwp_infer_e2_1",       SEEDS, BASE_SUBWP)
sthwp_noped_ex, sthwp_noped_n = agregar_seeds("sthwp_infer_e2_1_noped", SEEDS, BASE_STHWP)
subwp_noped_ex, subwp_noped_n = agregar_seeds("subwp_infer_e2_1_noped", SEEDS, BASE_SUBWP)

all_goals = sorted(set(sthwp_e21_ex) | set(subwp_e21_ex))

# ── SECCIÓN 1: Comparativa STHWP vs SUBWP en E2.1 por goal ──────────────────

print(f"\n{'='*80}")
print(f" SECCIÓN 1 — STHWP vs SUBWP: E2.1 (con peatón), por goal")
print(f" n = {len(SEEDS)} seeds × {N_EP} ep/goal = {len(SEEDS)*N_EP} ep por goal")
print(f"{'='*80}")
print(f"{'Goal':<10} {'STHWP%':>8} {'IC95%':>14} {'SUBWP%':>8} {'IC95%':>14} {'Δpp':>7} {'p-valor':>9} {'sig':>4}")
print("-"*80)

sig_goals_s1 = []
for gid in all_goals:
    k1 = sthwp_e21_ex.get(gid, 0); n1 = sthwp_e21_n.get(gid, 0)
    k2 = subwp_e21_ex.get(gid, 0); n2 = subwp_e21_n.get(gid, 0)
    if n1 == 0 or n2 == 0:
        continue
    p1 = k1/n1*100; p2 = k2/n2*100
    lo1, hi1 = wilson_ci(k1, n1)
    lo2, hi2 = wilson_ci(k2, n2)
    # Chi-cuadrado de homogeneidad (Fisher si celdas con 0)
    tabla = [[k1, n1-k1], [k2, n2-k2]]
    if min(k1, n1-k1, k2, n2-k2) == 0:
        _, pval = scipy_stats.fisher_exact(tabla)
    else:
        chi2, pval, _, _ = scipy_stats.chi2_contingency(tabla, correction=False)
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
    ci1 = f"[{lo1*100:.1f}-{hi1*100:.1f}]"
    ci2 = f"[{lo2*100:.1f}-{hi2*100:.1f}]"
    print(f"{gid:<10} {p1:>7.1f}% {ci1:>14} {p2:>7.1f}% {ci2:>14} {p1-p2:>+6.1f} {pval:>9.4f} {sig:>4}")
    if pval < 0.05:
        sig_goals_s1.append((gid, p1-p2, pval))

t1_sthwp, ex1, to1 = tasa_global(sthwp_e21_ex, sthwp_e21_n)
t1_subwp, ex2, to2 = tasa_global(subwp_e21_ex, subwp_e21_n)
lo_s, hi_s = wilson_ci(ex1, to1)
lo_w, hi_w = wilson_ci(ex2, to2)
print("-"*80)
print(f"{'GLOBAL':<10} {t1_sthwp:>7.1f}% [{lo_s*100:.1f}-{hi_s*100:.1f}]  {t1_subwp:>7.1f}% [{lo_w*100:.1f}-{hi_w*100:.1f}]  {t1_sthwp-t1_subwp:>+6.1f}")

if sig_goals_s1:
    print(f"\n  Goals con diferencia significativa (p<0.05): {len(sig_goals_s1)}")
    for gid, delta, pv in sig_goals_s1:
        favor = "STHWP" if delta > 0 else "SUBWP"
        print(f"    {gid}: Δ={delta:+.1f}pp → {favor} (p={pv:.4f})")

# ── SECCIÓN 2: Impacto del peatón por sistema ────────────────────────────────

print(f"\n{'='*80}")
print(f" SECCIÓN 2 — Impacto del peatón: E2.1_con vs E2.1_sin peatón")
print(f"{'='*80}")

noped_disponible = len(sthwp_noped_ex) > 0 or len(subwp_noped_ex) > 0

if not noped_disponible:
    print("  [!] CSVs de inferencia sin peatón no encontrados.")
    print("      Ejecuta primero: bash run_e2_1_noped.sh")
else:
    for label, con_ex, con_n, sin_ex, sin_n in [
        ("STHWP", sthwp_e21_ex, sthwp_e21_n, sthwp_noped_ex, sthwp_noped_n),
        ("SUBWP", subwp_e21_ex, subwp_e21_n, subwp_noped_ex, subwp_noped_n),
    ]:
        if not sin_ex:
            print(f"  {label}: datos sin peatón no disponibles aún")
            continue
        t_con,  ex_c, to_c = tasa_global(con_ex,  con_n)
        t_sin,  ex_s, to_s = tasa_global(sin_ex,  sin_n)
        lo_c, hi_c = wilson_ci(ex_c, to_c)
        lo_s2, hi_s2 = wilson_ci(ex_s, to_s)
        print(f"\n  {label}:")
        print(f"    Con peatón:   {t_con:.1f}%  IC95%=[{lo_c*100:.1f}–{hi_c*100:.1f}]")
        print(f"    Sin peatón:   {t_sin:.1f}%  IC95%=[{lo_s2*100:.1f}–{hi_s2*100:.1f}]")
        print(f"    Coste peatón: {t_con-t_sin:+.1f}pp")

        # Por goal: goals más afectados por el peatón
        impactos = []
        for gid in sorted(con_ex):
            kc = con_ex.get(gid, 0); nc = con_n.get(gid, 0)
            ks = sin_ex.get(gid, 0); ns = sin_n.get(gid, 0)
            if nc == 0 or ns == 0:
                continue
            delta = kc/nc*100 - ks/ns*100
            impactos.append((gid, delta))
        impactos.sort(key=lambda x: x[1])
        print(f"    Goals más perjudicados por peatón (Δ más negativo):")
        for gid, delta in impactos[:5]:
            print(f"      {gid}: {delta:+.1f}pp")
        print(f"    Goals con peatón beneficioso (replanning ayuda):")
        for gid, delta in impactos[-3:]:
            if delta > 0:
                print(f"      {gid}: {delta:+.1f}pp")

# ── SECCIÓN 3: Resumen global por experimento ────────────────────────────────

print(f"\n{'='*80}")
print(f" SECCIÓN 3 — Resumen global (IC 95% Wilson)")
print(f"{'='*80}")

experimentos = [
    ("STHWP E2.1 con peatón", sthwp_e21_ex,   sthwp_e21_n),
    ("SUBWP E2.1 con peatón", subwp_e21_ex,   subwp_e21_n),
]
if sthwp_noped_ex:
    experimentos.append(("STHWP E2.1 sin peatón", sthwp_noped_ex, sthwp_noped_n))
if subwp_noped_ex:
    experimentos.append(("SUBWP E2.1 sin peatón", subwp_noped_ex, subwp_noped_n))

print(f"{'Experimento':<28} {'Éxito%':>8} {'IC95% inf':>10} {'IC95% sup':>10} {'n':>6}")
print("-"*65)
for nombre, ex_d, n_d in experimentos:
    tasa, ex, to = tasa_global(ex_d, n_d)
    if to == 0:
        print(f"{nombre:<28}  No disponible")
        continue
    lo, hi = wilson_ci(ex, to)
    print(f"{nombre:<28} {tasa:>7.1f}%  {lo*100:>9.1f}%  {hi*100:>9.1f}% {to:>6}")

# ── SECCIÓN 4: Test global STHWP vs SUBWP ───────────────────────────────────

print(f"\n{'='*80}")
print(f" SECCIÓN 4 — Test global STHWP vs SUBWP (chi² sobre totales E2.1)")
print(f"{'='*80}")

ex_s = sum(sthwp_e21_ex.values()); n_s = sum(sthwp_e21_n.values())
ex_w = sum(subwp_e21_ex.values()); n_w = sum(subwp_e21_n.values())
tabla_global = [[ex_s, n_s - ex_s], [ex_w, n_w - ex_w]]
chi2_g, pval_g, _, _ = scipy_stats.chi2_contingency(tabla_global, correction=False)
# Cohen's h (effect size para proporciones)
p1 = ex_s / n_s; p2 = ex_w / n_w
h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

print(f"  STHWP: {ex_s}/{n_s} ({p1*100:.1f}%)")
print(f"  SUBWP: {ex_w}/{n_w} ({p2*100:.1f}%)")
print(f"  χ²={chi2_g:.2f}  p={pval_g:.6f}  Cohen's h={abs(h):.3f}")
sig_str = "SIGNIFICATIVO" if pval_g < 0.05 else "no significativo"
efecto = "pequeño" if abs(h) < 0.2 else ("mediano" if abs(h) < 0.5 else "grande")
print(f"  → Diferencia {sig_str} (p<0.05). Tamaño de efecto: {efecto} (h={abs(h):.3f})")

print(f"\n{'='*80}\n")
