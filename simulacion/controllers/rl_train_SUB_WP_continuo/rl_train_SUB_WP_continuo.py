"""
Entry point del controller rl_train_SUB_WP_continuo para Webots.

Delega al script de la etapa activa según el contenido de current_stage.txt:

  Valor              Script
  ─────────────────────────────────────────────────────────────────
  1_s42              train_stage1_s42.py
  1_s123             train_stage1_s123.py
  1_s524             train_stage1_s524.py
  2_s42              train_stage2_s42.py
  2_s123             train_stage2_s123.py
  2_s524             train_stage2_s524.py
  infer_2_s42        inferencia_subwp/infer_stage2_s42.py   ← mide headings reales
  infer_2_s123       inferencia_subwp/infer_stage2_s123.py
  infer_2_s524       inferencia_subwp/infer_stage2_s524.py
  3_s42              train_stage3_s42.py
  3_s123             train_stage3_s123.py
  3_s524             train_stage3_s524.py
  4_s42              train_stage4_s42.py
  4_s123             train_stage4_s123.py
  4_s524             train_stage4_s524.py
  infer_4_s42        inferencia_subwp/infer_stage4_s42.py
  infer_4_s123       inferencia_subwp/infer_stage4_s123.py
  infer_4_s524       inferencia_subwp/infer_stage4_s524.py
  5_s42              train_stage5_s42.py
  5_s123             train_stage5_s123.py
  5_s524             train_stage5_s524.py
  infer_5_s42        inferencia_subwp/infer_stage5_s42.py
  infer_5_s123       inferencia_subwp/infer_stage5_s123.py
  infer_5_s524       inferencia_subwp/infer_stage5_s524.py
  infer_ciclo_s42    inferencia_subwp/infer_ciclo_s42.py
  infer_ciclo_s123   inferencia_subwp/infer_ciclo_s123.py
  infer_ciclo_s524   inferencia_subwp/infer_ciclo_s524.py
"""

import os
import runpy

STAGE_FILE = os.path.join(os.path.dirname(__file__), "current_stage.txt")

with open(STAGE_FILE, "r") as f:
    stage = f.read().strip()

scripts = {
    # Stage 1 — approach goal_00
    "1_s42":   "train_stage1_s42.py",
    "1_s123":  "train_stage1_s123.py",
    "1_s524":  "train_stage1_s524.py",
    # Stage 2 — approach todos los goals
    "2_s42":   "train_stage2_s42.py",
    "2_s123":  "train_stage2_s123.py",
    "2_s524":  "train_stage2_s524.py",
    # Inferencia stage 1 (goal_01 solo)
    "infer_1_s42":  "inferencia_subwp/infer_stage1_s42.py",
    "infer_1_s123": "inferencia_subwp/infer_stage1_s123.py",
    "infer_1_s524": "inferencia_subwp/infer_stage1_s524.py",
    # Inferencia stage 2 (s42 mide headings, s123/s524 solo estadísticas)
    "infer_2_s42":  "inferencia_subwp/infer_stage2_s42.py",
    "infer_2_s123": "inferencia_subwp/infer_stage2_s123.py",
    "infer_2_s524": "inferencia_subwp/infer_stage2_s524.py",
    # Inferencia stage 3 (exit puro, 28 goals × 100 ep)
    "infer_3_s42":  "inferencia_subwp/infer_stage3_s42.py",
    "infer_3_s123": "inferencia_subwp/infer_stage3_s123.py",
    "infer_3_s524": "inferencia_subwp/infer_stage3_s524.py",
    # Stage 3 — exit puro con headings reales
    "3_s42":   "train_stage3_s42.py",
    "3_s123":  "train_stage3_s123.py",
    "3_s524":  "train_stage3_s524.py",
    # Stage 4 — ciclo completo
    "4_s42":   "train_stage4_s42.py",
    "4_s123":  "train_stage4_s123.py",
    "4_s524":  "train_stage4_s524.py",
    # Inferencia stage 4 (ciclo completo)
    "infer_4_s42":  "inferencia_subwp/infer_stage4_s42.py",
    "infer_4_s123": "inferencia_subwp/infer_stage4_s123.py",
    "infer_4_s524": "inferencia_subwp/infer_stage4_s524.py",
    # Stage 5 — return puro (descarga → espera)
    "5_s42":   "train_stage5_s42.py",
    "5_s123":  "train_stage5_s123.py",
    "5_s524":  "train_stage5_s524.py",
    # Inferencia stage 5 (return puro)
    "infer_5_s42":  "inferencia_subwp/infer_stage5_s42.py",
    "infer_5_s123": "inferencia_subwp/infer_stage5_s123.py",
    "infer_5_s524": "inferencia_subwp/infer_stage5_s524.py",
    # Inferencia ciclo completo (approach + exit + return, stage5_final, 100 ep/goal)
    "infer_ciclo_s42":  "inferencia_subwp/infer_ciclo_s42.py",
    "infer_ciclo_s123": "inferencia_subwp/infer_ciclo_s123.py",
    "infer_ciclo_s524": "inferencia_subwp/infer_ciclo_s524.py",
    # Inferencia con obstáculos dinámicos (peatones, 100 ep/goal)
    "infer_din_s123":   "inferencia_subwp/infer_dinamico_s123.py",
    "infer_din_s524":   "inferencia_subwp/infer_dinamico_s524.py",
    # Inferencia dinámica retrained (modelos reentrenados con peatones)
    "infer_din2_s42":   "inferencia_subwp/infer_din_retrained_s42.py",
    "infer_din2_s123":  "inferencia_subwp/infer_din_retrained_s123.py",
    "infer_din2_s524":  "inferencia_subwp/infer_din_retrained_s524.py",
    # Stage 6 dinámico (fine-tuning con peatones, desde stage5_final)
    "6_din_s42":        "train_stage6_din_s42.py",
    "6_din_s123":       "train_stage6_din_s123.py",
    "6_din_s524":       "train_stage6_din_s524.py",
    # Stage 6 din v2 (Opción 1 + Opción A: ped pos aleatoria + obs peatones)
    "6_din_v2_s42":      "train_stage6_din_v2_s42.py",
    "6_din_v2_s123":     "train_stage6_din_v2_s123.py",
    "6_din_v2_s524":     "train_stage6_din_v2_s524.py",
    # Inferencia dinámica v2
    "infer_din_v2_s42":  "inferencia_subwp/infer_din_v2_s42.py",
    "infer_din_v2_s123": "inferencia_subwp/infer_din_v2_s123.py",
    "infer_din_v2_s524": "inferencia_subwp/infer_din_v2_s524.py",
    # Stage 6 din v2 cont (continuación 3M steps desde din_v2_final)
    "6_din_v2_cont_s42":  "train_stage6_din_v2_cont_s42.py",
    "6_din_v2_cont_s123": "train_stage6_din_v2_cont_s123.py",
    "6_din_v2_cont_s524": "train_stage6_din_v2_cont_s524.py",
    # Inferencia dinámica v2_cont
    "infer_din_v2_cont_s42":  "inferencia_subwp/infer_din_v2_cont_s42.py",
    "infer_din_v2_cont_s123": "inferencia_subwp/infer_din_v2_cont_s123.py",
    "infer_din_v2_cont_s524": "inferencia_subwp/infer_din_v2_cont_s524.py",
    # Ablación peatones fijos (v2_cont model, fixed ped positions)
    "infer_din_v2_cont_fixedped_s42":  "inferencia_subwp/infer_din_v2_cont_fixedped_s42.py",
    "infer_din_v2_cont_fixedped_s123": "inferencia_subwp/infer_din_v2_cont_fixedped_s123.py",
    "infer_din_v2_cont_fixedped_s524": "inferencia_subwp/infer_din_v2_cont_fixedped_s524.py",
    # Stage 6 din v3 (recompensa B1: penalización proximidad peatón)
    "6_din_v3_s42":  "train_stage6_din_v3_s42.py",
    "6_din_v3_s123": "train_stage6_din_v3_s123.py",
    "6_din_v3_s524": "train_stage6_din_v3_s524.py",
    # Inferencia dinámica v3
    "infer_din_v3_s42":  "inferencia_subwp/infer_din_v3_s42.py",
    "infer_din_v3_s123": "inferencia_subwp/infer_din_v3_s123.py",
    "infer_din_v3_s524": "inferencia_subwp/infer_din_v3_s524.py",
    # diagnóstico: mejor modelo v3 con peatones completamente estáticos
    "infer_din_v3_s42_staticped": "inferencia_subwp/infer_din_v3_s42_staticped.py",
    # diagnóstico: modelo v1 estático (ped_obs=False) con peatones físicamente presentes
    "infer_subwp_s42_s5_pedpresent": "inferencia_subwp/infer_subwp_s42_stage5_pedpresent.py",
    # replanning — v3 dinámico + replanificación A* dinámica (3 seeds)
    "infer_din_v3_s42_replanning":  "inferencia_subwp/infer_din_v3_s42_replanning.py",
    "infer_din_v3_s123_replanning": "inferencia_subwp/infer_din_v3_s123_replanning.py",
    "infer_din_v3_s524_replanning": "inferencia_subwp/infer_din_v3_s524_replanning.py",
    # replanning — v1 estático + peatones congelados + replanificación (3 seeds)
    "infer_subwp_s42_s5_static_replanning":  "inferencia_subwp/infer_subwp_s42_stage5_replanning.py",
    "infer_subwp_s123_s5_static_replanning": "inferencia_subwp/infer_subwp_s123_stage5_replanning.py",
    "infer_subwp_s524_s5_static_replanning": "inferencia_subwp/infer_subwp_s524_stage5_replanning.py",
    # Stage 6 din v4 (replanning integrado en entorno, desde v3_final)
    "6_din_v4_s42":  "train_stage6_din_v4_s42.py",
    "6_din_v4_s123": "train_stage6_din_v4_s123.py",
    "6_din_v4_s524": "train_stage6_din_v4_s524.py",
    # Inferencia dinámica v4
    "infer_din_v4_s42":  "inferencia_subwp/infer_din_v4_s42.py",
    "infer_din_v4_s123": "inferencia_subwp/infer_din_v4_s123.py",
    "infer_din_v4_s524": "inferencia_subwp/infer_din_v4_s524.py",
    # Inferencia v4 con logging de replanning por fase
    "infer_din_v4_s42_replanlog":  "inferencia_subwp/infer_din_v4_s42_replanlog.py",
    "infer_din_v4_s123_replanlog": "inferencia_subwp/infer_din_v4_s123_replanlog.py",
    "infer_din_v4_s524_replanlog": "inferencia_subwp/infer_din_v4_s524_replanlog.py",
    # ── r2: reentrenamiento con dropoff corregido (-11.0, 0.0) ──────────────
    # Stage 3 r2 — exit puro (desde stage2_final)
    "3_r2_s42":  "train_stage3_r2_s42.py",
    "3_r2_s123": "train_stage3_r2_s123.py",
    "3_r2_s524": "train_stage3_r2_s524.py",
    # Stage 4 r2 — approach+exit (desde stage3_r2_final)
    "4_r2_s42":  "train_stage4_r2_s42.py",
    "4_r2_s123": "train_stage4_r2_s123.py",
    "4_r2_s524": "train_stage4_r2_s524.py",
    # Stage 5 r2 — return puro (desde stage4_r2_final)
    "5_r2_s42":  "train_stage5_r2_s42.py",
    "5_r2_s123": "train_stage5_r2_s123.py",
    "5_r2_s524": "train_stage5_r2_s524.py",
    # Stage 6 din v3 r2 — peatones + B1 (desde stage5_r2_final)
    "6_din_v3_r2_s42":  "train_stage6_din_v3_r2_s42.py",
    "6_din_v3_r2_s123": "train_stage6_din_v3_r2_s123.py",
    "6_din_v3_r2_s524": "train_stage6_din_v3_r2_s524.py",
    # Stage 6 din v4 r2 — replanning integrado (desde stage6_din_v3_r2_final)
    "6_din_v4_r2_s42":  "train_stage6_din_v4_r2_s42.py",
    "6_din_v4_r2_s123": "train_stage6_din_v4_r2_s123.py",
    "6_din_v4_r2_s524": "train_stage6_din_v4_r2_s524.py",
    # Inferencia r2 — stage 3 (exit puro)
    "infer_3_r2_s42":  "inferencia_subwp/infer_stage3_r2_s42.py",
    "infer_3_r2_s123": "inferencia_subwp/infer_stage3_r2_s123.py",
    "infer_3_r2_s524": "inferencia_subwp/infer_stage3_r2_s524.py",
    # Inferencia r2 — stage 4 (approach+exit)
    "infer_4_r2_s42":  "inferencia_subwp/infer_stage4_r2_s42.py",
    "infer_4_r2_s123": "inferencia_subwp/infer_stage4_r2_s123.py",
    "infer_4_r2_s524": "inferencia_subwp/infer_stage4_r2_s524.py",
    # Inferencia r2 — stage 5 (return puro)
    "infer_5_r2_s42":  "inferencia_subwp/infer_stage5_r2_s42.py",
    "infer_5_r2_s123": "inferencia_subwp/infer_stage5_r2_s123.py",
    "infer_5_r2_s524": "inferencia_subwp/infer_stage5_r2_s524.py",
    # Inferencia r2 — din v3 (ciclo completo + peatones)
    "infer_din_v3_r2_s42":  "inferencia_subwp/infer_din_v3_r2_s42.py",
    "infer_din_v3_r2_s123": "inferencia_subwp/infer_din_v3_r2_s123.py",
    "infer_din_v3_r2_s524": "inferencia_subwp/infer_din_v3_r2_s524.py",
    # Inferencia r2 — din v4 (ciclo completo + peatones + replanning)
    "infer_din_v4_r2_s42":  "inferencia_subwp/infer_din_v4_r2_s42.py",
    "infer_din_v4_r2_s123": "inferencia_subwp/infer_din_v4_r2_s123.py",
    "infer_din_v4_r2_s524": "inferencia_subwp/infer_din_v4_r2_s524.py",
    # Inferencia r2 — din v4 SIN peatones físicos (world noped, modelo 48-dim intacto)
    "infer_din_v4_r2_noped_s42":  "inferencia_subwp/infer_din_v4_r2_noped_s42.py",
    "infer_din_v4_r2_noped_s123": "inferencia_subwp/infer_din_v4_r2_noped_s123.py",
    "infer_din_v4_r2_noped_s524": "inferencia_subwp/infer_din_v4_r2_noped_s524.py",
    # ── EXPERIMENTOS (carpeta experimentos/) ─────────────────────────────────
    # E1 — 1 peatón (solo PEDESTRIAN_1, trayectoria actual x∈[-2,4] y=0.3)
    "e1_1ped_s42":  "experimentos/scripts/subwp_e1_1ped_s42.py",
    "e1_1ped_s123": "experimentos/scripts/subwp_e1_1ped_s123.py",
    "e1_1ped_s524": "experimentos/scripts/subwp_e1_1ped_s524.py",
    "infer_e1_1ped_s42":  "experimentos/scripts/subwp_infer_e1_1ped_s42.py",
    "infer_e1_1ped_s123": "experimentos/scripts/subwp_infer_e1_1ped_s123.py",
    "infer_e1_1ped_s524": "experimentos/scripts/subwp_infer_e1_1ped_s524.py",
    # E1.2 — sesgo 70% goals 19-22 + 24-28 (clusters A y B) desde modelos E1
    "e1_2_s42":  "experimentos/scripts/subwp_e1_2_s42.py",
    "e1_2_s123": "experimentos/scripts/subwp_e1_2_s123.py",
    "e1_2_s524": "experimentos/scripts/subwp_e1_2_s524.py",
    "infer_e1_2_s42":  "experimentos/scripts/subwp_infer_e1_2_s42.py",
    "infer_e1_2_s123": "experimentos/scripts/subwp_infer_e1_2_s123.py",
    "infer_e1_2_s524": "experimentos/scripts/subwp_infer_e1_2_s524.py",
    # E1.3 — trayectoria P1 extendida x∈[-4,4] (fine-tune desde E1.2, 48 dims)
    "e1_3_s42":  "experimentos/scripts/subwp_e1_3_s42.py",
    "e1_3_s123": "experimentos/scripts/subwp_e1_3_s123.py",
    "e1_3_s524": "experimentos/scripts/subwp_e1_3_s524.py",
    "infer_e1_3_s42":  "experimentos/scripts/subwp_infer_e1_3_s42.py",
    "infer_e1_3_s123": "experimentos/scripts/subwp_infer_e1_3_s123.py",
    "infer_e1_3_s524": "experimentos/scripts/subwp_infer_e1_3_s524.py",
    # E1.4 — patience reward + replan agresivo (fine-tune desde E1.3, 48 dims)
    "e1_4_s42":  "experimentos/scripts/subwp_e1_4_s42.py",
    "e1_4_s123": "experimentos/scripts/subwp_e1_4_s123.py",
    "e1_4_s524": "experimentos/scripts/subwp_e1_4_s524.py",
    "infer_e1_4_s42":  "experimentos/scripts/subwp_infer_e1_4_s42.py",
    "infer_e1_4_s123": "experimentos/scripts/subwp_infer_e1_4_s123.py",
    "infer_e1_4_s524": "experimentos/scripts/subwp_infer_e1_4_s524.py",
    # E1.5 — exit-corridor reward + curriculum ponderado (fine-tune desde E1.4, 48 dims)
    "e1_5_s42":  "experimentos/scripts/subwp_e1_5_s42.py",
    "e1_5_s123": "experimentos/scripts/subwp_e1_5_s123.py",
    "e1_5_s524": "experimentos/scripts/subwp_e1_5_s524.py",
    "infer_e1_5_s42":  "experimentos/scripts/subwp_infer_e1_5_s42.py",
    "infer_e1_5_s123": "experimentos/scripts/subwp_infer_e1_5_s123.py",
    "infer_e1_5_s524": "experimentos/scripts/subwp_infer_e1_5_s524.py",
    # E2.0 — Pre-entrenamiento stage6 sin peatón (base para E2.1)
    "e2_0_s42":  "experimentos/scripts/subwp_e2_0_s42.py",
    "e2_0_s123": "experimentos/scripts/subwp_e2_0_s123.py",
    "e2_0_s524": "experimentos/scripts/subwp_e2_0_s524.py",
    # E2.0b — Corrección s123: base stage4 (stage5 tenía olvido catastrófico)
    "e2_0b_s123": "experimentos/scripts/subwp_e2_0b_s123.py",
    # E2.1 — Obs realista: solo LIDAR 5m (sin supervisor), 40 dims
    "e2_1_s42":  "experimentos/scripts/subwp_e2_1_s42.py",
    "e2_1_s123": "experimentos/scripts/subwp_e2_1_s123.py",
    "e2_1_s524": "experimentos/scripts/subwp_e2_1_s524.py",
    "infer_e2_1_s42":  "experimentos/scripts/subwp_infer_e2_1_s42.py",
    "infer_e2_1_s123": "experimentos/scripts/subwp_infer_e2_1_s123.py",
    "infer_e2_1_s524": "experimentos/scripts/subwp_infer_e2_1_s524.py",
    # E2.2 — Replanning LIDAR dinámico (sin supervisor), fine-tune desde E2.1
    "e2_2_s42":  "experimentos/scripts/subwp_e2_2_s42.py",
    "e2_2_s123": "experimentos/scripts/subwp_e2_2_s123.py",
    "e2_2_s524": "experimentos/scripts/subwp_e2_2_s524.py",
    "infer_e2_2_s42":  "experimentos/scripts/subwp_infer_e2_2_s42.py",
    "infer_e2_2_s123": "experimentos/scripts/subwp_infer_e2_2_s123.py",
    "infer_e2_2_s524": "experimentos/scripts/subwp_infer_e2_2_s524.py",
    # E2.1 sin peatón — ablación: modelo E2.1 en mundo sin peatón (cuantifica impacto dinámico)
    "infer_e2_1_noped_s42":  "experimentos/scripts/subwp_infer_e2_1_noped_s42.py",
    "infer_e2_1_noped_s123": "experimentos/scripts/subwp_infer_e2_1_noped_s123.py",
    "infer_e2_1_noped_s524": "experimentos/scripts/subwp_infer_e2_1_noped_s524.py",
    # E2.2b — Fix LIDAR_MIN_DYNAMIC 1.5→2.5m (filtrar paredes pasillos estrechos), fine-tune desde E2.1
    "e2_2b_s42":  "experimentos/scripts/subwp_e2_2b_s42.py",
    "e2_2b_s123": "experimentos/scripts/subwp_e2_2b_s123.py",
    "e2_2b_s524": "experimentos/scripts/subwp_e2_2b_s524.py",
    "infer_e2_2b_s42":  "experimentos/scripts/subwp_infer_e2_2b_s42.py",
    "infer_e2_2b_s123": "experimentos/scripts/subwp_infer_e2_2b_s123.py",
    "infer_e2_2b_s524": "experimentos/scripts/subwp_infer_e2_2b_s524.py",
    # E1_pred — obs predictiva P1 (48→52 dims, weight transplant desde E1)
    "e1_pred_s42":  "experimentos/scripts/subwp_e1_pred_s42.py",
    "e1_pred_s123": "experimentos/scripts/subwp_e1_pred_s123.py",
    "e1_pred_s524": "experimentos/scripts/subwp_e1_pred_s524.py",
    "infer_e1_pred_s42":  "experimentos/scripts/subwp_infer_e1_pred_s42.py",
    "infer_e1_pred_s123": "experimentos/scripts/subwp_infer_e1_pred_s123.py",
    "infer_e1_pred_s524": "experimentos/scripts/subwp_infer_e1_pred_s524.py",
    # Evaluación de generalización WH02-05, fase 1 sin peatón (modelo E2.1)
    # parametrizado por env vars EVAL_WH / EVAL_SEED / EVAL_OUT_CSV (ver runner)
    "eval_gen": "experimentos/scripts/subwp_eval_generalizacion.py",
    # E2.1-R0 — mismo modelo E2.1, peatón activo, replanificación LIDAR
    # desactivada (env.enable_dynamic_replanning=False). Ver E2_1_R0_README.md.
    "eval_gen_r0": "experimentos/scripts/subwp_eval_generalizacion_e2_1_r0.py",
}

if stage not in scripts:
    raise ValueError(
        f"current_stage.txt contiene '{stage}'. "
        f"Valores válidos: {list(scripts.keys())}"
    )

script_path = os.path.join(os.path.dirname(__file__), scripts[stage])
print(f"[rl_train_SUB_WP_continuo] Lanzando etapa '{stage}' → {scripts[stage]}")
runpy.run_path(script_path, run_name="__main__")
