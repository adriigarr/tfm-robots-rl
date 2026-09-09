"""
Entry point del controller rl_train_STHWP para Webots.

Webots requiere que el script principal se llame igual que la carpeta del
controller. Este fichero delega al script de la etapa activa según el
contenido de current_stage.txt:

  Valor          Script
  ─────────────────────────────────────────
  1_s42          train_stage1_s42.py    (stage 1, seed=42)
  1_s123         train_stage1_s123.py   (stage 1, seed=123)
  1_s524         train_stage1_s524.py   (stage 1, seed=524)
  2_s42          train_stage2_s42.py         (stage 2, seed=42)
  2_s123         train_stage2_s123.py        (stage 2, seed=123)
  2_s524         train_stage2_s524.py        (stage 2, seed=524)
  3_s42          train_stage3_s42.py         (stage 3, seed=42)
  3_s123         train_stage3_s123.py        (stage 3, seed=123)
  3_s524         train_stage3_s524.py        (stage 3, seed=524)
  4              train_stage4.py
  infer_2_s42    inferencia_sthwp/infer_stage2_s42.py
  infer_2_s123   inferencia_sthwp/infer_stage2_s123.py
  infer_2_s524   inferencia_sthwp/infer_stage2_s524.py

Para cambiar de etapa basta con editar current_stage.txt y relanzar Webots.
"""

import os
import runpy

STAGE_FILE = os.path.join(os.path.dirname(__file__), "current_stage.txt")

with open(STAGE_FILE, "r") as f:
    stage = f.read().strip()

scripts = {
    "1_s42":   "train_stage1_s42.py",
    "1_s123":  "train_stage1_s123.py",
    "1_s524":  "train_stage1_s524.py",
    "2_s42":      "train_stage2_s42.py",
    "2_s123":     "train_stage2_s123.py",
    "2_s524":     "train_stage2_s524.py",
    "3_s42":      "train_stage3_s42.py",
    "3_s123":     "train_stage3_s123.py",
    "3_s524":     "train_stage3_s524.py",
    "3v2_s42":    "train_stage3v2_s42.py",      # i6 (headings reales, reward marcha atrás)
    "3v2_s123":   "train_stage3v2_s123.py",     # i6
    "3v2_s524":   "train_stage3_s524.py",       # i6 (train_stage3_s524.py ya actualizado)
    "4":          "train_stage4.py",
    "infer_s1":   "inferencia_sthwp/infer_stage1.py",
    "infer_2_s42":  "inferencia_sthwp/infer_stage2_s42.py",
    "infer_2_s123": "inferencia_sthwp/infer_stage2_s123.py",
    "infer_2_s524": "inferencia_sthwp/infer_stage2_s524.py",
    "demo_2":       "inferencia_sthwp/demo_stage2.py",
    # run002 — inferencia completa por stage
    "infer_r2_s524_s1": "inferencia_sthwp/infer_run002_s524_stage1.py",
    "infer_r2_s524_s2": "inferencia_sthwp/infer_run002_s524_stage2.py",
    "infer_r2_s524_s3": "inferencia_sthwp/infer_run002_s524_stage3.py",
    "infer_r2_s524_s4": "inferencia_sthwp/infer_run002_s524_stage4.py",
    # run002 — inferencia stage 3 v2 (headings reales + reward marcha atrás)
    "infer_r2_s42_s3v2":  "inferencia_sthwp/infer_run002_s42_stage3v2.py",
    "infer_r2_s123_s3v2": "inferencia_sthwp/infer_run002_s123_stage3v2.py",
    "infer_r2_s524_s3v2": "inferencia_sthwp/infer_run002_s524_stage3v2.py",
    # run002 — stage 4 v2 (entrenado desde stage 3 v2)
    "4v2_s42":            "train_stage4v2_s42.py",
    "4v2_s123":           "train_stage4v2_s123.py",
    "4v2_s524":           "train_stage4v2_s524.py",
    "infer_r2_s42_s4v2":  "inferencia_sthwp/infer_run002_s42_stage4v2.py",
    "infer_r2_s123_s4v2": "inferencia_sthwp/infer_run002_s123_stage4v2.py",
    "infer_r2_s524_s4v2": "inferencia_sthwp/infer_run002_s524_stage4v2.py",
    # run002 — stage 5 (retorno puro, entrenado desde stage 4 v2)
    "5_s42":             "train_stage5_s42.py",
    "5_s123":            "train_stage5_s123.py",
    "5_s524":            "train_stage5_s524.py",
    "infer_r2_s42_s5":   "inferencia_sthwp/infer_run002_s42_stage5.py",
    "infer_r2_s123_s5":  "inferencia_sthwp/infer_run002_s123_stage5.py",
    "infer_r2_s524_s5":  "inferencia_sthwp/infer_run002_s524_stage5.py",
    # run002 — inferencia ciclo completo (ap+exit encadenado con retorno)
    "infer_r2_s42_cc":   "inferencia_sthwp/infer_run002_s42_ciclo_completo.py",
    "infer_r2_s123_cc":  "inferencia_sthwp/infer_run002_s123_ciclo_completo.py",
    "infer_r2_s524_cc":  "inferencia_sthwp/infer_run002_s524_ciclo_completo.py",
    # run002 — inferencia con obstáculos dinámicos (peatones, 50 ep/goal)
    "infer_din_s42":     "inferencia_sthwp/infer_dinamico_s42.py",
    "infer_din_s524":    "inferencia_sthwp/infer_dinamico_s524.py",
    # run003 — stage 6 estático (ciclo completo, un único modelo, sin peatones)
    "6_s42":             "train_stage6_s42.py",
    "6_s123":            "train_stage6_s123.py",
    "6_s524":            "train_stage6_s524.py",
    # run003 — inferencia stage 6 estático (ciclo completo, modelo unificado)
    "infer_r3_s42_s6":   "inferencia_sthwp/infer_run003_s42_stage6.py",
    "infer_r3_s123_s6":  "inferencia_sthwp/infer_run003_s123_stage6.py",
    "infer_r3_s524_s6":  "inferencia_sthwp/infer_run003_s524_stage6.py",
    # run003 — inferencia stage 6 dinámico (modelos reentrenados con peatones)
    "infer_r3_s42_s6din":  "inferencia_sthwp/infer_run003_s42_stage6_din.py",
    "infer_r3_s123_s6din": "inferencia_sthwp/infer_run003_s123_stage6_din.py",
    "infer_r3_s524_s6din": "inferencia_sthwp/infer_run003_s524_stage6_din.py",
    # run003 — stage 6 dinámico (fine-tuning con peatones)
    "6_din_s42":         "train_stage6_din_s42.py",
    "6_din_s123":        "train_stage6_din_s123.py",
    "6_din_s524":        "train_stage6_din_s524.py",
    # run003 — stage 6 din v2 (Opción 1 + Opción A: ped pos aleatoria + obs peatones)
    "6_din_v2_s42":          "train_stage6_din_v2_s42.py",
    "6_din_v2_s123":         "train_stage6_din_v2_s123.py",
    "6_din_v2_s524":         "train_stage6_din_v2_s524.py",
    # run003 — inferencia stage 6 din v2
    "infer_r3_s42_s6din_v2":  "inferencia_sthwp/infer_run003_s42_stage6_din_v2.py",
    "infer_r3_s123_s6din_v2": "inferencia_sthwp/infer_run003_s123_stage6_din_v2.py",
    "infer_r3_s524_s6din_v2": "inferencia_sthwp/infer_run003_s524_stage6_din_v2.py",
    # run003 — stage 6 din v2 cont (continuación 3M steps desde din_v2_final)
    "6_din_v2_cont_s42":      "train_stage6_din_v2_cont_s42.py",
    "6_din_v2_cont_s123":     "train_stage6_din_v2_cont_s123.py",
    "6_din_v2_cont_s524":     "train_stage6_din_v2_cont_s524.py",
    # run003 — inferencia stage 6 din v2 cont
    "infer_r3_s42_s6din_v2_cont":  "inferencia_sthwp/infer_run003_s42_stage6_din_v2_cont.py",
    "infer_r3_s123_s6din_v2_cont": "inferencia_sthwp/infer_run003_s123_stage6_din_v2_cont.py",
    "infer_r3_s524_s6din_v2_cont": "inferencia_sthwp/infer_run003_s524_stage6_din_v2_cont.py",
    # run003 — ablación peatones fijos (v2_cont model, fixed ped positions)
    "infer_r3_s123_s6din_v2_cont_fixedped": "inferencia_sthwp/infer_run003_s123_stage6_din_v2_cont_fixedped.py",
    "infer_r3_s524_s6din_v2_cont_fixedped": "inferencia_sthwp/infer_run003_s524_stage6_din_v2_cont_fixedped.py",
    # run003 — stage 6 din v3 (recompensa B1: penalización proximidad peatón)
    "6_din_v3_s42":  "train_stage6_din_v3_s42.py",
    "6_din_v3_s123": "train_stage6_din_v3_s123.py",
    "6_din_v3_s524": "train_stage6_din_v3_s524.py",
    # run003 — inferencia stage 6 din v3
    "infer_r3_s42_s6din_v3":  "inferencia_sthwp/infer_run003_s42_stage6_din_v3.py",
    "infer_r3_s123_s6din_v3": "inferencia_sthwp/infer_run003_s123_stage6_din_v3.py",
    "infer_r3_s524_s6din_v3": "inferencia_sthwp/infer_run003_s524_stage6_din_v3.py",
    # demo movimiento peatones (sin modelo, robot estático)
    "demo_pedestrians": "inferencia_sthwp/demo_pedestrians.py",
    # diagnóstico: mejor modelo v3 con peatones completamente estáticos
    "infer_r3_s524_s6din_v3_staticped": "inferencia_sthwp/infer_run003_s524_stage6_din_v3_staticped.py",
    # diagnóstico: modelo v1 estático (ped_obs=False) con peatones físicamente presentes
    "infer_r3_s524_s6_pedpresent": "inferencia_sthwp/infer_run003_s524_stage6_static_pedpresent.py",
    # replanning — v3 dinámico + replanificación A* dinámica (3 seeds)
    "infer_r3_s42_s6din_v3_replanning":  "inferencia_sthwp/infer_run003_s42_stage6_din_v3_replanning.py",
    "infer_r3_s123_s6din_v3_replanning": "inferencia_sthwp/infer_run003_s123_stage6_din_v3_replanning.py",
    "infer_r3_s524_s6din_v3_replanning": "inferencia_sthwp/infer_run003_s524_stage6_din_v3_replanning.py",
    # replanning — v1 estático + peatones congelados + replanificación (3 seeds)
    "infer_r3_s42_s6_static_replanning":  "inferencia_sthwp/infer_run003_s42_stage6_static_replanning.py",
    "infer_r3_s123_s6_static_replanning": "inferencia_sthwp/infer_run003_s123_stage6_static_replanning.py",
    "infer_r3_s524_s6_static_replanning": "inferencia_sthwp/infer_run003_s524_stage6_static_replanning.py",
    # run003 — stage 6 din v4 (replanning integrado en entorno, desde v3_final)
    "6_din_v4_s42":  "train_stage6_din_v4_s42.py",
    "6_din_v4_s123": "train_stage6_din_v4_s123.py",
    "6_din_v4_s524": "train_stage6_din_v4_s524.py",
    # run003 — inferencia stage 6 din v4
    "infer_r3_s42_s6din_v4":  "inferencia_sthwp/infer_run003_s42_stage6_din_v4.py",
    "infer_r3_s123_s6din_v4": "inferencia_sthwp/infer_run003_s123_stage6_din_v4.py",
    "infer_r3_s524_s6din_v4": "inferencia_sthwp/infer_run003_s524_stage6_din_v4.py",
    # ── EXPERIMENTOS (carpeta experimentos/) ─────────────────────────────────
    # E1 — 1 peatón (solo PEDESTRIAN_1, trayectoria actual x∈[-2,4] y=0.3)
    "e1_1ped_s42":  "experimentos/scripts/sthwp_e1_1ped_s42.py",
    "e1_1ped_s123": "experimentos/scripts/sthwp_e1_1ped_s123.py",
    "e1_1ped_s524": "experimentos/scripts/sthwp_e1_1ped_s524.py",
    "infer_e1_1ped_s42":  "experimentos/scripts/sthwp_infer_e1_1ped_s42.py",
    "infer_e1_1ped_s123": "experimentos/scripts/sthwp_infer_e1_1ped_s123.py",
    "infer_e1_1ped_s524": "experimentos/scripts/sthwp_infer_e1_1ped_s524.py",
    # E1.2 — sesgo 70% goals 24-28 (sector P1) desde modelos E1
    "e1_2_s42":  "experimentos/scripts/sthwp_e1_2_s42.py",
    "e1_2_s123": "experimentos/scripts/sthwp_e1_2_s123.py",
    "e1_2_s524": "experimentos/scripts/sthwp_e1_2_s524.py",
    "infer_e1_2_s42":  "experimentos/scripts/sthwp_infer_e1_2_s42.py",
    "infer_e1_2_s123": "experimentos/scripts/sthwp_infer_e1_2_s123.py",
    "infer_e1_2_s524": "experimentos/scripts/sthwp_infer_e1_2_s524.py",
    # E1.3 — trayectoria P1 extendida x∈[-4,4] (fine-tune desde E1_pred, 52 dims)
    "e1_3_s42":  "experimentos/scripts/sthwp_e1_3_s42.py",
    "e1_3_s123": "experimentos/scripts/sthwp_e1_3_s123.py",
    "e1_3_s524": "experimentos/scripts/sthwp_e1_3_s524.py",
    "infer_e1_3_s42":  "experimentos/scripts/sthwp_infer_e1_3_s42.py",
    "infer_e1_3_s123": "experimentos/scripts/sthwp_infer_e1_3_s123.py",
    "infer_e1_3_s524": "experimentos/scripts/sthwp_infer_e1_3_s524.py",
    # E1.4 — patience reward + proximidad reforzada (fine-tune desde E1.3, 52 dims)
    "e1_4_s42":  "experimentos/scripts/sthwp_e1_4_s42.py",
    "e1_4_s123": "experimentos/scripts/sthwp_e1_4_s123.py",
    "e1_4_s524": "experimentos/scripts/sthwp_e1_4_s524.py",
    "infer_e1_4_s42":  "experimentos/scripts/sthwp_infer_e1_4_s42.py",
    "infer_e1_4_s123": "experimentos/scripts/sthwp_infer_e1_4_s123.py",
    "infer_e1_4_s524": "experimentos/scripts/sthwp_infer_e1_4_s524.py",
    # E1.5 — exit-corridor reward + curriculum ponderado (fine-tune desde E1.4, 52 dims)
    "e1_5_s42":  "experimentos/scripts/sthwp_e1_5_s42.py",
    "e1_5_s123": "experimentos/scripts/sthwp_e1_5_s123.py",
    "e1_5_s524": "experimentos/scripts/sthwp_e1_5_s524.py",
    "infer_e1_5_s42":  "experimentos/scripts/sthwp_infer_e1_5_s42.py",
    "infer_e1_5_s123": "experimentos/scripts/sthwp_infer_e1_5_s123.py",
    "infer_e1_5_s524": "experimentos/scripts/sthwp_infer_e1_5_s524.py",
    # E2.1 — Obs realista: solo LIDAR 5m (sin supervisor), 40 dims
    "e2_1_s42":  "experimentos/scripts/sthwp_e2_1_s42.py",
    "e2_1_s123": "experimentos/scripts/sthwp_e2_1_s123.py",
    "e2_1_s524": "experimentos/scripts/sthwp_e2_1_s524.py",
    "infer_e2_1_s42":  "experimentos/scripts/sthwp_infer_e2_1_s42.py",
    "infer_e2_1_s123": "experimentos/scripts/sthwp_infer_e2_1_s123.py",
    "infer_e2_1_s524": "experimentos/scripts/sthwp_infer_e2_1_s524.py",
    # E2.2 — Replanning LIDAR dinámico (sin supervisor), fine-tune desde E2.1
    "e2_2_s42":  "experimentos/scripts/sthwp_e2_2_s42.py",
    "e2_2_s123": "experimentos/scripts/sthwp_e2_2_s123.py",
    "e2_2_s524": "experimentos/scripts/sthwp_e2_2_s524.py",
    "infer_e2_2_s42":  "experimentos/scripts/sthwp_infer_e2_2_s42.py",
    "infer_e2_2_s123": "experimentos/scripts/sthwp_infer_e2_2_s123.py",
    "infer_e2_2_s524": "experimentos/scripts/sthwp_infer_e2_2_s524.py",
    # E2.1 sin peatón — ablación: modelo E2.1 en mundo sin peatón (cuantifica impacto dinámico)
    "infer_e2_1_noped_s42":  "experimentos/scripts/sthwp_infer_e2_1_noped_s42.py",
    "infer_e2_1_noped_s123": "experimentos/scripts/sthwp_infer_e2_1_noped_s123.py",
    "infer_e2_1_noped_s524": "experimentos/scripts/sthwp_infer_e2_1_noped_s524.py",
    # E2.2b — Fix LIDAR_MIN_DYNAMIC 1.5→2.5m (filtrar paredes pasillos estrechos), fine-tune desde E2.1
    "e2_2b_s42":  "experimentos/scripts/sthwp_e2_2b_s42.py",
    "e2_2b_s123": "experimentos/scripts/sthwp_e2_2b_s123.py",
    "e2_2b_s524": "experimentos/scripts/sthwp_e2_2b_s524.py",
    "infer_e2_2b_s42":  "experimentos/scripts/sthwp_infer_e2_2b_s42.py",
    "infer_e2_2b_s123": "experimentos/scripts/sthwp_infer_e2_2b_s123.py",
    "infer_e2_2b_s524": "experimentos/scripts/sthwp_infer_e2_2b_s524.py",
    # E1_pred — obs predictiva P1 (48→52 dims, weight transplant desde E1)
    "e1_pred_s42":  "experimentos/scripts/sthwp_e1_pred_s42.py",
    "e1_pred_s123": "experimentos/scripts/sthwp_e1_pred_s123.py",
    "e1_pred_s524": "experimentos/scripts/sthwp_e1_pred_s524.py",
    "infer_e1_pred_s42":  "experimentos/scripts/sthwp_infer_e1_pred_s42.py",
    "infer_e1_pred_s123": "experimentos/scripts/sthwp_infer_e1_pred_s123.py",
    "infer_e1_pred_s524": "experimentos/scripts/sthwp_infer_e1_pred_s524.py",
    # Evaluación de generalización WH02-05, fase 1 sin peatón (modelo E2.1)
    # parametrizado por env vars EVAL_WH / EVAL_SEED / EVAL_OUT_CSV (ver runner)
    "eval_gen": "experimentos/scripts/sthwp_eval_generalizacion.py",
    # E2.1-R0 — mismo modelo E2.1, peatón activo, replanificación LIDAR
    # desactivada (env.enable_dynamic_replanning=False). Ver E2_1_R0_README.md.
    "eval_gen_r0": "experimentos/scripts/sthwp_eval_generalizacion_e2_1_r0.py",
}

if stage not in scripts:
    raise ValueError(
        f"current_stage.txt contiene '{stage}'. "
        f"Valores válidos: {list(scripts.keys())}"
    )

script_path = os.path.join(os.path.dirname(__file__), scripts[stage])
print(f"[rl_train_STHWP] Lanzando etapa '{stage}' → {scripts[stage]}")
runpy.run_path(script_path, run_name="__main__")
