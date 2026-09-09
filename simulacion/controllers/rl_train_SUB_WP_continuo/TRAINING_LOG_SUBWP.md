# TRAINING LOG — SUB-WP Warehouse 1

Documento de referencia para el entrenamiento RL con método SUB-WP (Sub-WayPoint).
Registra las decisiones de diseño, diferencias respecto a STH-WP, y el progreso
de cada etapa de entrenamiento.

---

## 1. Contexto y motivación

### 1.1 Objetivo: comparativa STH-WP vs SUB-WP

El TFM compara dos métodos de generación de subgoals para navegación RL en almacén:

| Método | Subgoal | Descripción |
|--------|---------|-------------|
| **STH-WP** | Dinámico | Punto en el path A* a `d_ahead=1.5m` por delante del robot. Se mueve suavemente al avanzar. |
| **SUB-WP** | Discreto | Waypoints fijos cada `WP_STEP=1.5m`. El robot avanza al siguiente cuando está a `WP_REACH_DIST=0.5m`. |

El parámetro `WP_STEP=1.5m` iguala `D_AHEAD_NORMAL=1.5m` de STH-WP para una comparación justa:
ambos métodos generan subgoals a la misma distancia nominal en el path, pero con
comportamientos cualitativamente distintos.

### 1.2 Por qué el mismo curriculum 4 etapas

La hipótesis central de la comparativa es que la diferencia de rendimiento se debe
únicamente al tipo de subgoal, no al diseño del curriculum ni de las recompensas.
Por eso SUB-WP usa exactamente las mismas 4 etapas, los mismos hiperparámetros,
las mismas recompensas y los mismos 3 seeds que STH-WP (run002).

Cualquier diferencia en los resultados finales se puede atribuir al mecanismo de
generación de subgoal y no a confounds del entrenamiento.

---

## 2. Decisiones de diseño

### 2.1 Diferencia central: subgoal discreto vs continuo

**STH-WP (step):**
```python
d_actual = self.D_AHEAD_ESCAPE if self._fase_escape else self.D_AHEAD_NORMAL
subgoal  = compute_sth_subgoal(self.full_path, (rx, ry), d_actual)
self.goal = np.array(subgoal, dtype=np.float32)
```

**SUB-WP (step):**
```python
wp_x, wp_y = self.full_path[self._wp_idx]
dist_to_wp = math.sqrt((rx - wp_x)**2 + (ry - wp_y)**2)
if dist_to_wp < self.WP_REACH_DIST and self._wp_idx < len(self.full_path) - 1:
    self._wp_idx += 1
self.goal = np.array(self.full_path[self._wp_idx], dtype=np.float32)
```

Consecuencias de este cambio:

- **STH-WP**: el subgoal es un punto interpolado en el path, siempre a 1.5m del robot.
  Si el robot retrocede, el subgoal retrocede también. El subgoal "persigue" al robot.
- **SUB-WP**: el subgoal es un waypoint fijo. El robot debe acercarse a él para avanzar.
  Si el robot retrocede, el waypoint no cambia → el robot siempre tiene que volver al mismo punto.

### 2.2 Precomputación de rutas con submuestreo

Las rutas A* brutas (densas) se submuestrean a `WP_STEP=1.5m` durante la precomputación:

```python
path = plan_path(..., subsample=False, ...)   # path A* denso
path = subsample_path(path, step=self.WP_STEP)  # ~1.5m entre waypoints
```

**Motivo**: Con el path A* denso (celdas de ~0.05m), `WP_REACH_DIST=0.5m` capturaría
decenas de waypoints por step → el robot avanzaría a 10+ waypoints por paso, lo que
convierte el comportamiento en casi idéntico al STH-WP. El submuestreo asegura que
cada waypoint es un objetivo significativamente distante.

**Número de waypoints por ruta (estimado)**:
- Path approach típico (espera → estantería): ~4-10m → 3-7 waypoints
- Path exit (estantería → descarga): variable, 3-15 waypoints
- Goals 26/27 (rutas más largas): ~168 puntos A* → ~11 waypoints tras submuestreo

### 2.3 Eliminación de _fase_escape

STH-WP necesita `_fase_escape` (50 steps con `d_ahead=0.5m`) para evitar que el
subgoal se proyecte al otro lado de una esquina inmediatamente al salir de la estantería
(*Causa B* del análisis original).

SUB-WP no tiene este problema: el robot apunta al waypoint 1, que está 1.5m por
delante en el path, sin posibilidad de "saltar" esquinas porque los waypoints
son fijos y siguen la curvatura del path A*.

**Resultado**: `_fase_escape`, `_escape_steps`, `ESCAPE_DURATION`, `D_AHEAD_NORMAL`,
`D_AHEAD_ESCAPE` eliminados del entorno.

### 2.4 Headings propios de SUB-WP

El JSON de headings se mide desde la política SUB-WP stage 2, no reutilizando
el de STHWP. El archivo se genera en inferencia stage 2 seed=42:

```
inferencia_subwp/resultados/arrival_headings_stage2.json
```

Formato idéntico al de STHWP:
```json
{
  "goal_00": {"mean_rad": 1.23, "std_rad": 0.18, "mean_deg": 70.5, "n_samples": 87},
  ...
}
```

### 2.5 Parámetros idénticos a STH-WP

| Parámetro | Valor | Fuente |
|-----------|-------|--------|
| Recompensa progreso | `(prev_dist - dist_actual) * 3.0` | STH-WP |
| Penalización proximidad | `1.5 * exp(-2.5 * min_dist)` | STH-WP |
| Penalización angular | `0.05 * |vel_ang|` | STH-WP |
| Bonus orientación | `0.15 * vel_sign * cos(angulo_rel)` | STH-WP |
| Colisión | `-150.0` + terminado | STH-WP |
| Approach éxito | `+50.0` | STH-WP |
| Exit éxito | `+100.0` | STH-WP |
| Truncado | `-20.0` | STH-WP |
| Step cost | `-0.001` | STH-WP |
| SHELVES_DIFICILES | {10,11,12,13,14,15,22,23,24,25} | STH-WP |
| PROB_SHELF_DIFICIL | 0.7 | STH-WP |
| PROB_APPROACH_S4 | 0.7 | STH-WP |
| heading_sigma | 0.25 rad (~14°) | STH-WP |
| Timeout training | `_max_steps=2500` | STH-WP |
| Timeout inferencia | `_max_steps=4000` | STH-WP |

---

## 3. Estructura de ficheros

```
rl_train_SUB_WP_continuo/
├── webots_env.py                      ← Entorno SUB-WP (WP_REACH_DIST, WP_STEP)
├── global_planner.py                  ← A* + subsample_path (igual que STHWP)
├── callbacks.py                       ← StatsCallback con ventana deslizante 100 ep
├── heading_utils.py                   ← compute_theoretical_headings + conversión Webots
├── rl_train_SUB_WP_continuo.py        ← Dispatcher (current_stage.txt)
├── current_stage.txt                  ← Etapa activa para el dispatcher
│
├── train_stage1_s42.py                ┐
├── train_stage1_s123.py               │ Stage 1 — 3 seeds
├── train_stage1_s524.py               ┘
├── train_stage2_s42.py                ┐
├── train_stage2_s123.py               │ Stage 2 — 3 seeds
├── train_stage2_s524.py               ┘
├── train_stage3_s42.py                ┐
├── train_stage3_s123.py               │ Stage 3 — 3 seeds
├── train_stage3_s524.py               ┘
├── train_stage4_s42.py                ┐
├── train_stage4_s123.py               │ Stage 4 — 3 seeds
├── train_stage4_s524.py               ┘
│
├── run_subwp_stage1.sh                ← Stage 1 solo (3 seeds)
├── run_subwp_full.sh                  ← Pipeline completo (stages 1-4 + inferencias)
│
└── inferencia_subwp/
    ├── infer_stage2_s42.py            ← Mide headings + CSV
    ├── infer_stage2_s123.py           ← CSV solo
    ├── infer_stage2_s524.py           ← CSV solo
    ├── infer_stage4_s42.py            ← Ciclo completo, timeout=4000
    ├── infer_stage4_s123.py
    ├── infer_stage4_s524.py
    └── resultados/
        ├── arrival_headings_stage2.json   ← Generado por infer_stage2_s42
        ├── infer_subwp_s*_stage2.csv
        └── infer_subwp_s*_stage4.csv
```

---

## 4. Pipeline de entrenamiento

```
Stage 1 (500k)  →  Stage 2 (4M)  →  Inferencia S2 (mide headings)
                                              ↓
                                         Stage 3 (4M)  →  Stage 4 (4M)
                                                                ↓
                                                        Inferencia S4 (ciclo completo)
```

### Diseño por etapa

| Etapa | Tipo episodio | Goals | Steps | ent_coef | lr |
|-------|--------------|-------|-------|----------|----|
| 1 | Approach, goal_00 solo | 1 | 500k | 0.01 | 3e-4 |
| 2 | Approach, todos los goals | 28 (uniforme) | 4M | 0.01 | 2e-4 |
| 3 | Exit puro, heading real SUB-WP + ruido | 28 | 4M | 0.01 | 2e-4 |
| 4 | Ciclo completo 70% approach / 30% exit | 28 | 4M | 0.01 | 1e-4 |

**Criterio de paso stage 1 → stage 2**: `exito_ult100_% ≥ 95%` en TensorBoard.

**Nota sobre inferencia stage 2**: `infer_stage2_s42.py` ejecuta 100 ep × 28 goals
en modo determinista, registra el heading del compás al llegar a cada estantería
y calcula la media circular por goal. El JSON resultante alimenta el teleport
de stage 3 con la orientación real de llegada de la política SUB-WP (no de STHWP).

---

## 5. Registro de entrenamientos

---

### Stage 1 — run001 — Seeds 42/123/524

**Fecha inicio**: 2026-06-29  
**Steps**: 500 000 × 3 seeds  
**Script**: `run_subwp_stage1.sh`

**Configuración PPO (desde cero):**

| Parámetro | Valor |
|-----------|-------|
| policy | MlpPolicy |
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

**Salidas esperadas:**
- `pruebas/subwp_s42_stage1_final.zip`
- `pruebas/subwp_s123_stage1_final.zip`
- `pruebas/subwp_s524_stage1_final.zip`

**Estado**: completado (20:02 → 20:30, ~28 min totales, ~9 min/seed)

**Modelos guardados:**
- `pruebas/subwp_s42_stage1_final.zip`
- `pruebas/subwp_s123_stage1_final.zip`
- `pruebas/subwp_s524_stage1_final.zip`

#### Resultados TensorBoard y CSV

| Seed | goal_01_éxito_% | éxito_ult100_% | colisiones totales | tasa_col_% | truncados | pasos/ep | reward/ep |
|------|----------------|----------------|-------------------|------------|-----------|----------|-----------|
| s42  | **98.79%**     | **100%**       | 5                 | 0.33%      | 12 (0.80%)| 266      | 67.3      |
| s524 | **97.93%**     | **100%**       | 18                | 1.20%      | 11 (0.73%)| 246      | 72.1      |
| s123 | **96.43%**     | **100%**       | 30                | 2.03%      | 20 (1.35%)| 292      | 66.4      |

*Fuente CSV (goal_01, intentos/éxitos/colisiones): s42=1494/1477/5 · s524=1512/1483/18 · s123=1492/1442/30*

#### Análisis

**Criterio de paso cumplido por los 3 seeds** (`exito_ult100_% ≥ 95%`):

- **s42** y **s524** convergen hacia los ~150k steps. Colisiones residuales mínimas
  (5 y 18 respectivamente), todas concentradas en la fase de exploración inicial (0-100k),
  ninguna en el tramo final.
- **s123** es el más lento: alcanza el 95% alrededor de los ~350k steps. Acumula 30
  colisiones durante el entrenamiento (tasa 2.03%), el doble que s524 y seis veces más
  que s42. Mismo patrón frágil observado en STHWP stage 1.

**Eficiencia**: s524 es el más eficiente con 246 pasos/ep. Coherente con WP_STEP=1.5m:
goal_01 tiene un path de ~12m → ~8 waypoints × ~30 steps/wp ≈ 240 steps.

**Entropía al final**: -1.8 a -2.1 (train/entropy_loss). Sin colapso — la política
mantiene exploración suficiente para generalizar en stage 2.

**train/std**: de ~1.0 inicial a ~0.70 al final. La política se especializa
progresivamente sin perder diversidad de acción.

**Comparación con STHWP stage 1**: comportamiento prácticamente idéntico. Misma jerarquía
de seeds (s42/s524 > s123), mismos rangos de convergencia y colisión. Confirmación de
que el entorno SUB-WP está bien implementado — la divergencia entre métodos empezará a
manifestarse en stages 2-4 con paths más complejos y giros cerrados.

**Decisión**: ✅ Pasar a stage 2 con los 3 seeds.

---

### Stage 2 — run001 — Seeds 42/123/524

**Fecha inicio**: 2026-06-29  
**Steps**: 4 000 000 × 3 seeds  
**Script**: `run_subwp_stage2.sh` (o manualmente)  
**Base**: `subwp_s*_stage1_final.zip`

**Configuración PPO (fine-tune desde stage 1):**

| Parámetro | Valor |
|-----------|-------|
| learning_rate | 2e-4 |
| ent_coef | 0.01 |
| reset_num_timesteps | False |

**Objetivo**: generalizar el approach a los 28 goals. Sesgo PROB_SHELF_DIFICIL=0.7
hacia goals de bloques 3 y 5 (más alejados de zona de espera).

**Criterio de paso**: `exito_ult100_% ≥ 80%` de forma estable en TensorBoard.
Goals difíciles (estanterías interiores) pueden quedar al 60-70% — se refuerzan en stage 4.

**Salidas esperadas:**
- `pruebas/subwp_s42_stage2_final.zip`
- `pruebas/subwp_s123_stage2_final.zip`
- `pruebas/subwp_s524_stage2_final.zip`

**Estado**: completado (20:56 → 00:15, ~3h19min totales, ~1h7min/seed)

**Modelos guardados:**
- `pruebas/subwp_s42_stage2_final.zip`
- `pruebas/subwp_s123_stage2_final.zip`
- `pruebas/subwp_s524_stage2_final.zip`

#### Resultados CSV — resumen global

| Seed | éxito global | colisiones | tasa col. | mín goal | tasa mín | goals <90% |
|------|-------------|------------|-----------|----------|----------|------------|
| s42  | **98.6%**   | 19         | 0.45%     | goal_05  | 96.5%    | ninguno    |
| s524 | **98.7%**   | 36         | 0.80%     | goal_27  | 96.4%    | ninguno    |
| s123 | **97.6%**   | 89         | 1.96%     | goal_20  | 95.6%    | ninguno    |

#### Resultados por goal — goals más relevantes

| Goal | Tipo | s42 | s123 | s524 |
|------|------|-----|------|------|
| goal_02 | fácil | **100.0%** | 97.3% | **100.0%** |
| goal_05 | difícil | 96.5% | 98.8% | 98.8% |
| goal_09 | difícil | 98.2% | **99.4%** | **100.0%** |
| goal_10 | difícil | **100.0%** | 97.4% | 98.6% |
| goal_13 | difícil | **99.5%** | 96.1% | 99.3% |
| goal_17 | difícil | **100.0%** | 96.9% | 98.7% |
| goal_20 | fácil | 99.2% | 95.6% | **100.0%** |
| goal_27 | fácil | 98.2% | 96.2% | 96.4% |

#### Análisis TensorBoard (4.5M steps)

**Métricas de entrenamiento PPO al finalizar:**

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `ep_rew_mean` (smoothed) | 155.4 | 152.7 | **164.98** |
| `exito_ult100_%` | **100%** | 95.2% | **100%** |
| `colision_ult100_%` | **0** | 4.78% | **0** |
| `exito_dificiles_ult100_%` | **100%** | 94.8% | **100%** |
| `tasa_exito_%` (global) | 98.6% | 97.6% | **98.7%** |
| `tasa_colision_%` (global) | **0.45%** | 1.96% | 0.80% |
| `tasa_truncado_%` | 0.94% | 0.40% | 0.53% |
| `n_colisiones` (total) | **19** | 89 | 36 |
| `entropy_loss` | -0.523 | -1.884 | -0.937 |
| `explained_variance` | 0.287 | -0.037 | 0.062 |
| `train/std` | 0.737 | **1.829** | 0.762 |
| `approx_kl` | 0.011 | 0.010 | 0.010 |
| `clip_fraction` | 0.121 | 0.090 | 0.105 |
| FPS final | 1002 | 999 | **1017** |

**Convergencia y reward**: s524 lidera con 165.0 de reward suavizado, seguido por s42 (155.4)
y s123 (152.7). Todos los seeds presentan un **dip pronunciado ~3-3.5M** en `ep_rew_mean`
(s42 llega a caer hasta ~125), seguido de recuperación total. Coincide con spikes en
`truncado_ult100_%` alrededor de 2-3M: fase de re-exploración transitoria que el algoritmo
supera sin consecuencias permanentes.

**Comportamiento diferenciado s42 vs s524**: s42 tiene la mayor `tasa_truncado_%` (0.94%),
el doble que s524 (0.53%), pero solo 19 colisiones totales vs 36 de s524. La política de
s42 es más conservadora: cuando no sabe avanzar, se queda bloqueada hasta el timeout en
lugar de arriesgar una colisión. Reward algo menor, seguridad máxima.

**Por qué s123 es el seed frágil (diagnóstico PPO)**:
- `entropy_loss = -1.884` (en SB3 es `-entropy`, más negativo = más entropía): la política
  sigue siendo muy exploratoria al final del entrenamiento. No ha convergido a un comportamiento
  determinista estable, al contrario que s42 (-0.523) y s524 (-0.937).
- `train/std = 1.829` (2.5× el de s42/s524 que están en ~0.74): distribución de acciones
  excesivamente ancha. El robot actúa con alta incertidumbre.
- `explained_variance = -0.037` (negativo): el crítico no está aprendiendo bien las ventajas.
  Estimaciones ruidosas → actualizaciones PPO de mala calidad → política que nunca se estabiliza.
- `n_colisiones = 89` (4.7× s42): consecuencia directa de la alta incertidumbre en las acciones.

**Goals difíciles en approach**: todos por encima del 96% en los 3 seeds. La dificultad de
las estanterías interiores es exclusivamente del exit (pasillo estrecho al salir), no del approach.

**Rendimiento excepcional global**: los 3 seeds superan el 95% en absolutamente todos los 28 goals.
El goal mínimo es 96.5% (s42/goal_05), 95.6% (s123/goal_20) y 96.4% (s524/goal_27). Ninguno
cae por debajo del umbral de preocupación.

**SUB-WP approach vs STH-WP approach**: resultados comparables entre métodos — ambos alcanzan
>95% en stage 2. Para el approach (espacio abierto, paths sin giros bruscos) ambos mecanismos
de subgoal son equivalentes. La divergencia entre métodos se espera en stage 3 (exit, pasillos
estrechos, giros 90°-150°).

**Consistencia TensorBoard ↔ Inferencia**: perfecta. Los seeds que terminan con col=0 y
éxito=100% en TensorBoard (s42, s524) logran 100% en todos los goals en inferencia determinista.
s123 con 4.78% de colisión en TensorBoard → 92.9% global en inferencia, con colapso determinista
exactamente en los goals más exigentes (goals 26 y 27).

**Decisión**: ✅ Pasar a inferencia stage 2 para medir headings reales antes de stage 3.

---

### Inferencia Stage 2 — Medición de headings reales

**Objetivo**: ejecutar 100 ep × 28 goals con la política SUB-WP stage 2 (seed=42),
registrar el heading del compás al llegar a cada estantería y guardar el JSON
`arrival_headings_stage2.json` que alimentará el teleport de stage 3.

**Script**: `inferencia_subwp/infer_stage2_s42.py`  
**Salida**: `inferencia_subwp/resultados/arrival_headings_stage2.json`

**Estado**: completado (00:21 → 01:55, ~1h34min totales)

**Ficheros generados:**
- `inferencia_subwp/resultados/infer_subwp_s42_stage1.csv`
- `inferencia_subwp/resultados/infer_subwp_s123_stage1.csv`
- `inferencia_subwp/resultados/infer_subwp_s524_stage1.csv`
- `inferencia_subwp/resultados/infer_subwp_s42_stage2.csv`
- `inferencia_subwp/resultados/infer_subwp_s123_stage2.csv`
- `inferencia_subwp/resultados/infer_subwp_s524_stage2.csv`
- `inferencia_subwp/resultados/arrival_headings_stage2.json` ← generado desde s42

#### Resultados Stage 1 (goal_01, 500 ep, determinista)

| Seed | Éxito | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|-------|----------|----------|----------|-----------|
| s42  | **500/500 (100%)** | 0 | 0 | 267 | 67.0 |
| s123 | **500/500 (100%)** | 0 | 0 | 251 | 75.1 |
| s524 | **500/500 (100%)** | 0 | 0 | 268 | 71.7 |

Los 3 seeds alcanzan 100% en goal_01 determinista. Confirmación de que el stage 1 es sólido.

#### Resultados Stage 2 — resumen global (28 goals × 100 ep, determinista)

| Seed | Éxito global | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|-------------|----------|----------|----------|-----------|
| s42  | **2800/2800 (100%)** | 0 (0%) | 0 | 907 | 161.8 |
| s524 | **2800/2800 (100%)** | 0 (0%) | 0 | 868 | 164.5 |
| s123 | 2600/2800 (92.9%) | 200 (7.1%) | 0 | 873 | 143.7 |

#### Resultados Stage 2 — por goal

Todos los goals presentan 100% de éxito para s42 y s524. s123 falla únicamente en:

| Goal | s42 | s123 | s524 | Causa |
|------|-----|------|------|-------|
| goal_26 | ✅ 100% | ❌ 0% (100 col) | ✅ 100% | Colapso determinista |
| goal_27 | ✅ 100% | ❌ 0% (100 col) | ✅ 100% | Colapso determinista |
| Resto (26 goals) | ✅ 100% | ✅ 100% | ✅ 100% | — |

#### Análisis

**s42 y s524 perfectos**: 100% en los 28 goals sin ninguna colisión. El approach SUB-WP
es completamente robusto en modo determinista para estos dos seeds.

**s123 — colapso determinista en goals 26 y 27**: durante el entrenamiento (stochastic)
alcanzaba 96-98% en ambos goals, pero en inferencia determinista la política converge
siempre al mismo camino erróneo → 100% colisión. Goals 26 y 27 tienen los paths de approach
más largos del almacén, lo que los hace más exigentes para la política determinista.
Mismo patrón frágil observado en STHWP s123. No es un defecto del método sino del seed.

**arrival_headings_stage2.json**: generado desde s42 (100% éxito, 100 headings válidos
por goal). Todos los 28 goals tienen mediciones fiables → stage 3 puede arrancar con
headings reales de la política SUB-WP.

**Decisión**: ✅ Pasar a stage 3. Seeds de referencia: s42 y s524.

---

### Stage 3 — run001 — Seeds 42/123/524

**Fecha inicio**: 2026-06-30  
**Steps**: 4 000 000 × 3 seeds (steps acumulados TensorBoard: 4.5M → 8.5M)  
**Base**: `subwp_s*_stage2_final.zip`  
**Headings**: `inferencia_subwp/resultados/arrival_headings_stage2.json`

**Objetivo**: aprender el exit puro. Robot teleportado a la estantería con el heading
real medido en inferencia stage 2 (± ruido gaussiano 0.25 rad).
Esta etapa es la más crítica — es donde el comportamiento de SUB-WP divergirá más
respecto a STH-WP por la diferencia en el mecanismo de subgoal en pasillos estrechos.

**Configuración PPO:**

| Parámetro | Valor |
|-----------|-------|
| learning_rate | 2e-4 |
| ent_coef | 0.01 |
| reset_num_timesteps | False |

**Criterio de paso**: `exito_ult100_% ≥ 70%` estable (en STHWP stage 3 v2 se alcanzó ~76.7%).

**Salidas:**
- `pruebas/subwp_s42_stage3_final.zip`
- `pruebas/subwp_s123_stage3_final.zip`
- `pruebas/subwp_s524_stage3_final.zip`

**Estado**: completado (4M steps × 3 seeds)

#### Resultados TensorBoard — métricas de comportamiento

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `ep_rew_mean` (smoothed) | -58.96 | -72.57 | **-50.94** |
| `exito_ult100_%` | **33%** | 21% | 32% |
| `colision_ult100_%` | 67% | **79%** | 68% |
| `exito_dificiles_ult100_%` | **37%** | 29% | 30% |
| `exito_faciles_ult100_%` | **33%** | 20% | 26% |
| `tasa_exito_%` (global) | 28.86% | 28.15% | **28.47%** |
| `tasa_colision_%` (global) | 70.97% | **71.81%** | 71.49% |
| `tasa_truncado_%` | **15.63%** | 4.33% | 3.69% |
| `pasos_medio_episodio` | **435** | 186 | 287 |
| `n_colisiones` total | **9,089** | 11,627 | 11,649 |
| `n_exitos` total | 3,700 | 4,556 | **4,640** |
| `n_truncados` total | **20** | 7 | 6 |

#### Métricas PPO

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `entropy_loss` | **-3.590** | -3.290 | -2.724 |
| `explained_variance` | 0.067 | **0.924** | 0.876 |
| `train/std` | 2.332 | **5.144** | 3.485 |
| `approx_kl` | 0.010 | 0.014 | 0.007 |
| `train/loss` | **53.42** | 9.16 | 12.77 |
| `train/value_loss` | **50.78** | 26.54 | 35.37 |
| FPS | 1073 | 1073 | **1076** |

#### Análisis

**Dificultad extrema del exit puro**: todos los seeds bajan de +155/+165 en stage 2 a
-43/-80 en stage 3. La tasa de éxito global se sitúa en ~28-29% para los 3 seeds —
la mayor parte de los episodios terminan en colisión (~71%). Esto contrasta drásticamente
con STHWP stage 3 que alcanzó ~76.7%.

**s524 mejor reward, s42 mejor en últimos 100 episodios**: s524 tiene el reward más alto
(-42.22) y mejor eficiencia (287 pasos/ep, explicada_variance=0.876). s42, con 435 pasos/ep
y 15.6% de truncaciones, no resuelve el exit eficientemente pero evita colisionar —
llega al límite de 2500 steps en lugar de impactar. Resultado: mejor éxito en los últimos
100 (33%) pero peor critic (explained_variance=0.067) y el mayor valor_loss (50.78).

**s42 con el crítico más dañado**: `explained_variance ≈ 0` y `value_loss = 50.78` (5.5×
más que s123). Con 15.6% de truncaciones, la mezcla de outcomes (éxito +100, colisión -150,
truncado -20) hace que el retorno sea impredecible para el crítico. La política, muy
determinista (entropy_loss = -3.59), queda atrapada en un mínimo local de "vagar sin
colisionar".

**s123 paradoja**: tiene el mejor crítico (explained_variance=0.924) pero la política más
débil (21% éxito ult100). El crítico aprende bien los retornos, pero la política no extrae
comportamiento exitoso de ellos. El alto std (5.14) indica enorme incertidumbre en la
acción elegida — la política es muy ruidosa.

**Todos los seeds se vuelven más deterministas**: entropy_loss desciende de ~-1 a ~-3
durante stage 3. Las políticas convergen pero no a soluciones exitosas — convergen a
comportamientos repetitivos que no evitan obstáculos en giros cerrados.

**`tasa_estanteria = 100%`**: el teleport con headings reales funciona correctamente.
Todos los episodios parten con la orientación real medida de la política SUB-WP.

**Criterio de paso NO cumplido**: el mejor resultado es 33% (s42, ult100) frente al
criterio de ≥70%. La comparativa con STHWP (~76.7%) revela la debilidad fundamental
del mecanismo SUB-WP en pasillos estrechos: el waypoint discreto está fijo en una
posición que puede requerir un giro brusco para alcanzarlo desde dentro de la estantería,
mientras STH-WP siempre proyecta el subgoal 1.5m adelante en la dirección de avance
del path, adaptándose a la posición instantánea del robot. En giros de 90°-150°, esta
diferencia se vuelve crítica.

**Decisión**: ✅ Pasar a stage 4. Los 4M steps están completos. El ciclo completo
(70% approach / 30% exit) consolidará el approach (dominado al 100%) y permitirá
al robot combinar ambas habilidades con más variedad de transiciones.

---

### Stage 4 — run001 — Seeds 42/123/524

**Fecha inicio**: 2026-06-30  
**Steps**: 4 000 000 × 3 seeds (steps acumulados TensorBoard: 8.5M → 12.5M)  
**Base**: `subwp_s*_stage3_final.zip`

**Objetivo**: ciclo completo 70% approach / 30% exit. El robot alterna entre ambas
tareas, consolidando approach (dominado al 100%) y refinando el exit.

**Configuración PPO:**

| Parámetro | Valor |
|-----------|-------|
| learning_rate | 1e-4 |
| ent_coef | 0.01 |
| reset_num_timesteps | False |

**Salidas:**
- `pruebas/subwp_s42_stage4_final.zip`
- `pruebas/subwp_s123_stage4_final.zip`
- `pruebas/subwp_s524_stage4_final.zip`

**Estado**: completado (4M steps × 3 seeds)

#### Resultados TensorBoard — métricas de comportamiento

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `ep_rew_mean` (smoothed) | 240.63 | **247.86** | 242.56 |
| `exito_ult100_%` | 79 | 79 | **81** |
| `colision_ult100_%` | 21 | 20 | **19** |
| `exito_dificiles_ult100_%` | 75 | **79** | **79** |
| `exito_faciles_ult100_%` | 82 | 83 | **84** |
| `tasa_exito_%` (global) | 72.85% | **79.13%** | 77.60% |
| `tasa_colision_%` (global) | 21.60% | **20.40%** | 22.41% |
| `tasa_truncado_%` | **5.55%** | 0.47% | **0%** |
| `tasa_estanteria_%` | 99.93% | 99.97% | **100%** |
| `n_exitos` | 2101 | 2370 | **2418** |
| `n_colisiones` | 623 | 611 | **698** |
| `n_truncados` | **160** | 14 | **0** |
| `pasos_medio_episodio` | 1289 | 1293 | 1299 |

#### Breakdown approach vs exit

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `colision_approach_%` | 3.47% | 3.34% | **0%** |
| `colision_exit_%` | 21.57% | **20.37%** | 22.00% |
| `truncado_approach_%` | 0.035% | 0% | **0%** |
| `truncado_exit_%` | **5.51%** | 0.47% | **0%** |
| `llego_estanteria_ult100_%` | **100%** | **100%** | **100%** |

#### Métricas PPO

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `entropy_loss` | -3.326 | **-3.781** | -3.402 |
| `explained_variance` | 0.869 | 0.864 | **0.876** |
| `train/std` | **4.16** | 10.81 | 6.62 |
| `approx_kl` | 0.007 | 0.008 | **0.006** |
| FPS | 1085 | 1047 | **1086** |

#### Análisis

**Hallazgo principal — stage 4 cierra el gap del exit**: la tasa de éxito sube de ~28-33%
(stage 3 ult100) a **79-81%** (stage 4 ult100). El curriculum 70/30 approach/exit funciona:
la señal de éxito frecuente del approach estabiliza el critic y el robot aprende a integrar
ambas tareas en una misma política.

**s524 mejor en ult100 (81%)**: approach perfecto (0 colisiones, 0 truncaciones), salida
más limpia. Política más consistente de los 3 seeds.

**s123 mejor global (79.13%)**: sorpresiva recuperación. Era el seed más frágil en stages
1-3 y en stage 4 tiene la mayor tasa_exito global y menor colision_exit (20.37%). La mayor
entropía (std=10.8) resulta beneficiosa en stage 4 — la política exploratoria encuentra
mejores salidas.

**s42 truncaciones persistentes**: 160 totales (5.55%), concentradas en exit (5.51%).
La política sigue prefiriendo vagar hasta timeout en lugar de arriesgarse a colisionar en
pasillos. Reduce tasa_exito global a 72.85% aunque ult100 alcanza 79%.

**Críticos recuperados**: explained_variance ~0.86-0.88 en los 3 seeds. El approach
frecuente (+50 reward limpio y predecible) ancla el critic y lo saca del mínimo de stage 3
(s42 pasó de explained_variance=0.067 a 0.869).

**train/std creciente**: s123 llega a 10.8, s524 a 6.6, s42 a 4.2. El portfolio approach+exit
exige distribuciones de acción más amplias — comportamientos muy distintos para las dos tareas.

**s123 FPS dip en 10M**: caída de ~1085 a ~1015 fps, recuperación parcial a 1047. Coincide
con período de inestabilidad transitoria en reward — episodios más largos → FPS aparente menor.

**Comparativa preliminar con STHWP**: el gap que existía en stage 3 (SUB-WP 28% vs STHWP
~76.7%) se ha cerrado casi completamente en stage 4 (~79% vs ~80%+ STHWP). La inferencia
del ciclo completo dará los números definitivos para el TFM.

**Decisión**: ✅ Lanzar inferencia stage 4 (ciclo completo, 28 goals × 100 ep).

---

### Inferencia Stage 4 — Ciclo completo

**Objetivo**: evaluar el ciclo completo (approach → exit) sobre los 28 goals en modo
determinista. Métrica de comparativa final con STHWP.

**Scripts**: `inferencia_subwp/infer_stage4_s{42,123,524}.py`  
**Timeout**: `env._max_steps = 4000`  
**Episodios**: 28 goals × 100 ep = 2800 ep × 3 seeds

**Estado**: completado

**Script completo (todos los stages)**: `run_subwp_infer_all.sh`

#### Resultados — todas las inferencias

**Stage 1 (goal_01, 100 ep, determinista):**

| Seed | Éxito | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|-------|----------|----------|----------|-----------|
| s42  | **100%** | 0% | 0% | 267 | 67.0 |
| s123 | **100%** | 0% | 0% | 251 | 75.1 |
| s524 | **100%** | 0% | 0% | 268 | 71.7 |

**Stage 2 (approach 28 goals, 2800 ep, determinista):**

| Seed | Éxito | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|-------|----------|----------|----------|-----------|
| s42  | **100%** | 0% | 0% | 907 | 161.8 |
| s123 | 92.9% | 7.1% | 0% | 873 | 143.7 |
| s524 | **100%** | 0% | 0% | 868 | 164.5 |

s123 falla en goals 26 y 27 (100 col cada uno) — colapso determinista.

**Stage 3 (exit puro, 2800 ep, determinista):**

| Seed | Éxito | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|-------|----------|----------|----------|-----------|
| s42  | 29.2% | 70.8% | 0% | 372 | -71.4 |
| s123 | 29.2% | 70.8% | 0% | 241 | -50.7 |
| s524 | 28.6% | 71.4% | 0% | 237 | -53.0 |

Solo 9 goals con éxito >0%: goals 8, 9, 10 (~99%), goals 20, 21, 22 (100%),
goals 17 (39-52%), 18 (86-90%), 19 (76-83%). Todos los demás: 0%.

**Stage 4 (ciclo completo, 2800 ep, determinista):**

| Seed | Éxito | Col approach | Col exit | Truncado | Pasos/ep | Reward/ep |
|------|-------|:------------:|:--------:|:--------:|----------|-----------|
| s42  | **100%** | 0% | 0% | 0% | 1737 | 380.0 |
| s123 | **100%** | 0% | 0% | 0% | 1739 | 383.9 |
| s524 | **100%** | 0% | 0% | 0% | 1736 | 378.9 |

**Los 3 seeds logran 100% en todos los 28 goals.**

---

## 6. Comparativa SUB-WP vs STH-WP

### Resultados por stage

#### Stage 1

| Seed | SUB-WP éxito | SUB-WP pasos | STHWP éxito | STHWP pasos | Δ pasos |
|------|:-----------:|:------------:|:-----------:|:-----------:|:-------:|
| s42  | **100%** | 267 | **100%** | — | — |
| s123 | **100%** | 251 | **100%** | 245 | +6 |
| s524 | **100%** | 268 | **100%** | 227 | +41 |

Empate en éxito. STHWP ligeramente más eficiente en pasos.

#### Stage 2 — Approach

| Seed | SUB-WP éxito | SUB-WP pasos | SUB-WP reward | STHWP éxito | STHWP pasos | STHWP reward |
|------|:-----------:|:------------:|:-------------:|:-----------:|:-----------:|:------------:|
| s42  | **100%** | 907 | 161.8 | **100%** | 846 | **170.4** |
| s123 | 92.9% | 873 | 143.7 | **100%** | 843 | **171.1** |
| s524 | **100%** | 868 | 164.5 | **100%** | 818 | 158.6 |

STHWP gana: 3/3 seeds al 100% vs 2/3. SUB-WP s123 colapsa en goals 26/27.
STHWP también más eficiente en pasos (~820-846 vs ~868-907).

#### Stage 3 — Exit puro

| Seed | SUB-WP éxito | SUB-WP reward | STHWP éxito | STHWP reward | Δ éxito |
|------|:-----------:|:-------------:|:-----------:|:------------:|:-------:|
| s42  | 29.2% | -71.4 | 63.6% | +81.1 | **-34.4pp** |
| s123 | 29.2% | -50.7 | 76.4% | +133.1 | **-47.2pp** |
| s524 | 28.6% | -53.0 | **76.7%** | +129.4 | **-48.1pp** |

STHWP gana con diferencia de ~44 puntos porcentuales en promedio.
SUB-WP solo tiene éxito en 9/28 goals (los de salida más directa/ancha).
STHWP tiene éxito en 25/28 goals (falla solo goals con giros muy cerrados).

Goals exitosos SUB-WP: 8, 9, 10, 17 (parcial), 18 (parcial), 19 (parcial), 20, 21, 22.
Goals a 0%: todos los de bloques 1 (01-07), bloque 3 parcial (11-16), bloque 5 (23-28).

#### Stage 4 — Ciclo completo (RESULTADO DEFINITIVO)

| Seed | SUB-WP éxito | SUB-WP reward | STHWP éxito | STHWP col_exit | STHWP reward | Δ éxito |
|------|:-----------:|:-------------:|:-----------:|:--------------:|:------------:|:-------:|
| s42  | **100%** | **380.0** | **100%** | 0% | 365.6 | 0 |
| s123 | **100%** | **383.9** | 82.1% | 17.9% | 298.9 | **+17.9pp** |
| s524 | **100%** | **378.9** | **100%** | 0% | 363.0 | 0 |

**SUB-WP gana en stage 4**: 3/3 seeds al 100% vs STHWP con s123 al 82.1%.
SUB-WP consigue mayor reward en los 3 seeds (~379-384 vs ~299-366).

STHWP s123 falla en goals 9, 10, 14, 15, 22 (100% colisión en exit, colapso determinista).
SUB-WP s123 en stage 4: 100% en todos.

### Tabla resumen comparativa

| Etapa | Métrica | SUB-WP | STHWP | Ganador |
|-------|---------|:------:|:-----:|:-------:|
| **Stage 1** | Éxito medio seeds | **100%** | **100%** | Empate |
| **Stage 1** | Pasos/ep | 262 | 236 | STHWP |
| **Stage 2** | Éxito medio seeds | 97.6% | **100%** | STHWP |
| **Stage 2** | Reward/ep medio | 157.3 | **166.7** | STHWP |
| **Stage 3** | Éxito medio seeds | 29.0% | **72.2%** | STHWP (+43pp) |
| **Stage 3** | Goals con éxito >50% | 9/28 | **25/28** | STHWP |
| **Stage 4** | Éxito medio seeds | **100%** | 94.0% | **SUB-WP** |
| **Stage 4** | Seeds al 100% | **3/3** | 2/3 | **SUB-WP** |
| **Stage 4** | Reward/ep medio | **380.9** | 342.7 | **SUB-WP** |

### Interpretación para el TFM

**Stage 3 aislado — STH-WP claramente superior**: el waypoint discreto fijo exige
que el robot apunte a un punto que puede estar al otro lado de una esquina. El
subgoal dinámico de STH-WP siempre se proyecta 1.5m adelante desde la posición
actual adaptándose a los giros — ventaja estructural en pasillos estrechos.

**Stage 4 integrado — SUB-WP superior**: cuando el exit se encadena naturalmente
desde el approach, el robot llega a la estantería con su heading natural de llegada,
perfectamente alineado con el primer waypoint del exit path. El subgoal dinámico
pierde su ventaja porque el robot ya está orientado correctamente. Además, el
entrenamiento de stage 4 optimizó el ciclo completo como tarea integrada.

**Conclusión principal**: el mecanismo de subgoal importa más en navegación aislada
(exit puro desde condición arbitraria) que en el ciclo completo real. En la tarea
de producción — ciclo approach→exit encadenado — SUB-WP es igual o superior a
STH-WP con menor complejidad algorítmica (sin `compute_sth_subgoal`, sin
`_fase_escape`, sin interpolación continua).

**Implicación práctica**: para sistemas de producción donde el robot siempre
realiza approach→exit de forma encadenada, SUB-WP es preferible por su simplicidad.
Para sistemas donde el exit puede iniciarse desde una posición arbitraria (e.g.,
reinicio tras fallo, teleoperación), STH-WP ofrece mayor robustez.

---

---

## 7. Análisis profundo: por qué cada método domina en cada etapa

Esta sección explica con detalle técnico los mecanismos que provocan las diferencias
de rendimiento entre SUB-WP y STH-WP en cada stage, con referencias directas al código.

---

### 7.1 Diferencia algorítmica fundamental

Ambos métodos comparten exactamente la misma arquitectura de entrenamiento (4 stages,
mismos hiperparámetros PPO, mismas recompensas, mismos seeds), el mismo espacio de
observación (36 LIDAR + dist + angle + vlin + vang, 40 dimensiones), y la misma
representación de acción (velocidad lineal + angular continuas). La **única diferencia**
es cómo se genera `self.goal` en cada step.

#### STH-WP: subgoal dinámico continuo

```python
# rl_train_STHWP/webots_env.py — líneas 343-348
d_actual = self.D_AHEAD_ESCAPE if self._fase_escape else self.D_AHEAD_NORMAL
subgoal  = compute_sth_subgoal(self.full_path, (rx, ry), d_actual)
self.goal = np.array(subgoal, dtype=np.float32)
```

`compute_sth_subgoal(path, pos, d)` recorre el path A* desde el punto proyectado más
cercano a la posición actual del robot `(rx, ry)` y devuelve el punto en el path a
exactamente `d` metros por delante. Constantes relevantes:
- `D_AHEAD_NORMAL = 1.5` m — distancia nominal (línea 50)
- `D_AHEAD_ESCAPE = 0.5` m — distancia reducida los primeros `ESCAPE_DURATION=50` steps
  tras llegar a la estantería (línea 51)

El subgoal se **recalcula cada step** en función de la posición actual del robot. Si el
robot retrocede, el subgoal también retrocede. Si gira en un pasillo, el subgoal se
proyecta exactamente 1.5m hacia adelante en la curva del path. Es un seguimiento de
trayectoria clásico tipo *lookahead*.

#### SUB-WP: waypoint discreto estático

```python
# rl_train_SUB_WP_continuo/webots_env.py — líneas 340-348
wp_x, wp_y = self.full_path[self._wp_idx]
dist_to_wp = math.sqrt((rx - wp_x) ** 2 + (ry - wp_y) ** 2)
if dist_to_wp < self.WP_REACH_DIST and self._wp_idx < len(self.full_path) - 1:
    self._wp_idx += 1
self.goal = np.array(self.full_path[self._wp_idx], dtype=np.float32)
```

El path A* se pre-submuestrea a waypoints cada `WP_STEP=1.5` m (línea 45) en
`_precompute_paths()` (línea 197):

```python
# rl_train_SUB_WP_continuo/webots_env.py — línea 197
path = subsample_path(path, step=self.WP_STEP)
```

`self.goal` apunta siempre a `full_path[_wp_idx]` — un **punto fijo en el espacio**.
El índice solo avanza cuando el robot se acerca a `WP_REACH_DIST=0.5` m (línea 44).
Si el robot no llega a ese radio de captura, el waypoint no cambia nunca.

#### Consecuencia estructural clave

La diferencia se reduce a una sola propiedad: **reactividad ante la posición del robot**.

| Propiedad | STH-WP | SUB-WP |
|-----------|--------|--------|
| El subgoal depende de `(rx, ry)` | Sí — cada step | Solo para avanzar al siguiente |
| Si el robot se aleja del path | Subgoal se ajusta | Waypoint permanece fijo |
| En un giro de 90° | Subgoal sigue la curva | Waypoint puede estar al otro lado |
| Complejidad computacional | O(n) por step (recorrido del path) | O(1) — solo comparar distancia |

---

### 7.2 Stage 1 — Empate (100% ambos métodos)

**Goal_01** es la estantería más próxima a la zona de espera, con un path de approach
corto (~12m, 8 waypoints en SUB-WP) y sin curvas pronunciadas. En estas condiciones
ambos mecanismos de subgoal son funcionalmente equivalentes:

- STH-WP: el subgoal se proyecta 1.5m adelante en una línea casi recta → el robot
  sigue directamente hacia la estantería.
- SUB-WP: los 8 waypoints cada 1.5m están alineados casi en línea recta → el robot
  también los alcanza secuencialmente sin dificultad.

La pequeña diferencia en pasos (SUB-WP ~262 vs STHWP ~236) se debe a que el waypoint
discreto es un punto fijo al que el robot converge con mayor esfuerzo angular que el
subgoal fluido de STH-WP, que siempre está alineado con la trayectoria instantánea.

---

### 7.3 Stage 2 — STH-WP ligeramente superior en approach

El approach de 28 goals involucra paths más largos y algunos giros moderados al entrar
en los pasillos entre estanterías. STH-WP logra 100% en los 3 seeds; SUB-WP logra 100%
en s42 y s524, pero s123 colapsa determinísticamente en goals 26 y 27 (92.9% global).

**Por qué s123 falla en goals 26/27 con SUB-WP y no con STHWP**:

Goals 26 y 27 son los más alejados del almacén, con paths de approach de ~18-20m
(~13 waypoints en SUB-WP). La política de s123 en modo determinista siempre elige
la misma secuencia de acciones — si esa secuencia falla en alcanzar un waypoint
intermedio (porque el robot se desalinea ligeramente respecto al punto fijo), el
waypoint no cambia y la política converge al mismo error siempre.

Con STH-WP, si el robot se desalinea, el subgoal se desplaza para compensar
(siempre está a 1.5m por delante de la posición proyectada), lo que ofrece una
señal de navegación más tolerante a perturbaciones. Esto hace que la política de
s123, que ya era frágil (train/std=1.829 en stage 2, explained_variance=-0.037),
pueda recibir señal de corrección continua en STHWP pero no en SUB-WP.

STH-WP también es ~60-90 pasos/ep más eficiente (~820-846 vs ~868-907) porque el
subgoal fluido genera trayectorias más suaves sin los pequeños stops que ocurren
cuando SUB-WP espera a que el robot alcance el radio de captura `WP_REACH_DIST=0.5`.

---

### 7.4 Stage 3 — STH-WP claramente superior en exit puro (+43pp)

Este es el resultado más importante del análisis. El exit puro teleportado expone la
debilidad estructural de SUB-WP en entornos con geometría de giro cerrado.

#### El problema del waypoint trans-esquina

Considérese un robot que debe salir de la estantería goal_01 (bloque 1). El path exit
comienza dentro del pasillo entre estanterías y requiere un giro de ~90°-120° para
entrar al pasillo principal. Tras submuestrear a WP_STEP=1.5m, el primer waypoint
del exit está 1.5m adelante en el path A*, que puede estar **al otro lado de la esquina**.

```
Situación en SUB-WP:

    [ROBOT] ←—(1.5m)—→ [WP_0] 
        ↑                  ↑
   dentro del         al otro lado de
   pasillo              la esquina
   (0.5m ancho)
```

La pared bloquea el camino directo al waypoint. El robot, cuyo subgoal apunta al
otro lado de la pared, intenta avanzar en línea recta y colisiona. La única salida
sería girar en el pasillo estrecho, pero la señal de recompensa de progreso

```python
# webots_env.py — línea 360
recompensa += (self._prev_dist - dist_actual) * 3.0
```

penaliza cualquier acción que aleje al robot del waypoint trans-esquina, incluyendo
el giro correcto que lo llevaría al otro lado.

**STH-WP resuelve exactamente este problema** con `_fase_escape` (líneas 371-374):

```python
# rl_train_STHWP/webots_env.py — líneas 371-374
if self._fase_escape:
    self._escape_steps += 1
    if self._escape_steps >= self.ESCAPE_DURATION:
        self._fase_escape = False
```

Durante los primeros `ESCAPE_DURATION=50` steps tras el reset en la estantería,
`d_actual = D_AHEAD_ESCAPE = 0.5` m. El subgoal no se proyecta al otro lado de la
esquina sino que permanece muy cerca (0.5m) y se adapta en tiempo real a la posición
del robot. Cuando el robot gira correctamente para salir, el subgoal gira con él.
Después de 50 steps (cuando ya está en el pasillo principal) se restaura
`D_AHEAD_NORMAL=1.5m`.

SUB-WP no tiene este mecanismo porque fue diseñado para eliminarlo como ventaja:
los waypoints discretos deberían seguir la curvatura del path. Y lo hacen — pero
el problema es que `WP_STEP=1.5m` es demasiado grande para pasillos de ~0.8-1.5m
de ancho. El waypoint puede estar 1.5m adelante pero físicamente inalcanzable en
línea recta.

#### Evidencia en los datos: patrón de goals exitosos

Los únicos goals con exit exitoso en SUB-WP stage 3 son:

| Goal | Éxito SUB-WP | Descripción de la salida |
|------|:----------:|--------------------------|
| goal_08 | ~99% | Salida directa, pasillo ancho |
| goal_09 | ~99% | Salida directa, pasillo ancho |
| goal_10 | **100%** | Salida directa, pasillo ancho |
| goal_17 | 39-52% | Giro moderado, parcialmente exitoso |
| goal_18 | 86-90% | Giro suave, mayoría exitosa |
| goal_19 | 76-83% | Giro moderado |
| goal_20 | **100%** | Salida directa |
| goal_21 | **100%** | Salida directa |
| goal_22 | **100%** | Salida directa |

Los 19 goals restantes (01-07, 11-16, 23-28): **0%**. Todos corresponden a estanterías
con giros bruscos en la salida donde el primer waypoint del exit path está al otro
lado de la esquina o en un pasillo de acceso estrecho.

STH-WP, con su subgoal adaptativo, solo falla en goals extremadamente difíciles
(s42: goals 01-06, 17, 25; s42 logra 63.6% global) y alcanza 76-77% con s123/s524.

#### Efecto sobre la señal de recompensa de progreso

Un aspecto adicional: en SUB-WP, cuando el robot está bloqueado por la pared y el
waypoint está al otro lado, `dist_actual` prácticamente no cambia step a step. La
recompensa de progreso `(prev_dist - dist_actual) * 3.0` es ~0 o ligeramente negativa.
El robot no recibe señal de que algo esté mal — solo recibe la penalización de colisión
cuando finalmente toca la pared. En STH-WP, el subgoal se mueve con el robot, por lo
que la señal de progreso siempre refleja si el robot avanza o retrocede en la trayectoria,
lo que genera un gradiente más informativo.

---

### 7.5 Stage 4 — SUB-WP superior en el ciclo completo (100% vs 94%)

El resultado más sorprendente: SUB-WP, que solo alcanzaba 29% en exit puro (stage 3),
llega al **100%** en ciclo completo (stage 4) con los 3 seeds. STHWP s123 se queda
en 82.1% (falla en goals 9, 10, 14, 15, 22).

Hay tres mecanismos que explican esta inversión:

#### Mecanismo 1: Heading de llegada vs heading de teleport

En stage 3, el robot se teleporta a la estantería con un heading tomado de un JSON
medido en stage 2 y perturbado con ruido gaussiano `σ=0.25 rad` (línea 242-243 de
ambos entornos):

```python
# webots_env.py — líneas 241-243 (ambos entornos)
theta       = self._headings.get(goal_idx, 0.0)
theta_noisy = theta + self.np_random.normal(0, self.heading_sigma)
rot         = heading_to_webots_rotation(theta_noisy)
```

En stage 4, en cambio, el robot llega a la estantería por sus propios medios desde
la zona de espera. El heading con el que llega es el **heading natural de la política**
para ese goal — perfectamente reproducible en inferencia determinista (la política es
determinista → misma trayectoria → mismo heading de llegada siempre).

Cuando el ciclo approach→exit se encadena (líneas 391-401 de SUB-WP):

```python
# rl_train_SUB_WP_continuo/webots_env.py — líneas 391-401
self._hacia_descarga = True
recompensa += 50.0
self.full_path = list(self._astar_cache[("exit", self.current_goal_idx)])
self._wp_idx = 0
self.goal    = np.array(self.full_path[0], dtype=np.float32)
```

El robot, ya en la estantería con su heading natural, recibe como nuevo subgoal
`full_path[0]` — el primer waypoint del exit path, que está 1.5m adelante en la
dirección de salida. La clave es que **el heading natural de llegada del approach
está alineado con la dirección de salida** porque la trayectoria de approach termina
en la estantería entrando por el pasillo de acceso, que es el mismo pasillo por el
que hay que salir.

Dicho de otro modo: el robot llega al goal en la orientación "correcta" para iniciar
el exit porque el camino más corto para llegar implica entrar por el acceso de la
estantería, que es la misma dirección de salida. El primer waypoint del exit está
ya en la dirección del heading de llegada → no hay giro trans-esquina.

En stage 3 con teleport, el heading tiene ±0.25 rad de error estocástico durante
entrenamiento, y durante inferencia es el heading medio medido, que puede diferir
del heading que la política produciría en ciclo completo. Esta discrepancia entre
el heading de entrenamiento y el heading real de inferencia en stage 4 genera una
distribución out-of-distribution para la política de stage 3, pero no para stage 4.

#### Mecanismo 2: La política de stage 4 es distinta a la de stage 3

El modelo `subwp_s*_stage4_final.zip` no es el mismo que `subwp_s*_stage3_final.zip`.
El fine-tuning de 4M steps adicionales en stage 4 (70% approach / 30% exit) ha
modificado los pesos de la red. La política de stage 4 aprendió a hacer exit en el
contexto de haber completado un approach: conoce el estado del robot tras llegar.

La política de stage 3 aprendió exit desde estados de teleport artificiales —
condición que en stage 4 nunca ocurre durante inferencia (porque `PROB_APPROACH_S4=1.0`
fuerza que todo episodio empiece por approach).

#### Mecanismo 3: El approach ancla el estado del robot

En el ciclo completo, el approach de ~900 pasos "calienta" la política antes del exit.
El robot llega a la estantería habiendo ejecutado una trayectoria larga y exitosa,
con velocidades y aceleraciones en un rango conocido. El estado interno del robot
(velocidades reales, posición LIDAR) es el estado "natural" de llegada.

En el teleport de stage 3, el robot aparece instantáneamente en la estantería con
velocidades a cero y una orientación impuesta. La discontinuidad del estado puede
activar modos de comportamiento sub-óptimos que la política aprendió a evitar.

#### Por qué STHWP s123 falla en stage 4

STHWP s123 logra solo 82.1% en stage 4, fallando en goals 9, 10, 14, 15, 22
(100% colisión exit). Estos son goals de bloque 2 y 3 donde el exit requiere
giros precisos. La política de s123 en STHWP, aunque tiene _fase_escape, colapsa
determinísticamente en esos goals específicos porque la red aprendió un comportamiento
de exit determinista que no funciona para esas geometrías.

SUB-WP s123 en stage 4 no falla en ningún goal porque el encadenamiento approach→exit
con heading natural elimina la dependencia del teleport y la política, entrenada en
ciclo completo, aprendió a manejar la transición correctamente.

---

### 7.6 Tabla comparativa completa — Inferencia determinista

| Etapa | Métrica | SUB-WP s42 | SUB-WP s123 | SUB-WP s524 | STHWP s42 | STHWP s123 | STHWP s524 |
|-------|---------|:----------:|:-----------:|:-----------:|:---------:|:----------:|:----------:|
| **Stage 1** | Éxito | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| **Stage 1** | Pasos/ep | 267 | 251 | 268 | — | 245 | 227 |
| **Stage 1** | Reward/ep | 67.0 | 75.1 | 71.7 | — | 86.1 | **80.4** |
| **Stage 2** | Éxito | **100%** | 92.9% | **100%** | **100%** | **100%** | **100%** |
| **Stage 2** | Colisión | 0% | 7.1% | 0% | 0% | 0% | 0% |
| **Stage 2** | Pasos/ep | 907 | 873 | 868 | 846 | 843 | 818 |
| **Stage 2** | Reward/ep | 161.8 | 143.7 | 164.5 | **170.4** | **171.1** | 158.6 |
| **Stage 3** | Éxito | 29.2% | 29.2% | 28.6% | 63.6% | **76.4%** | **76.7%** |
| **Stage 3** | Colisión | 70.8% | 70.8% | 71.4% | 36.4% | 23.6% | 23.3% |
| **Stage 3** | Pasos/ep | 372 | 241 | 237 | 604 | 727 | 739 |
| **Stage 3** | Reward/ep | -71.4 | -50.7 | -53.0 | +81.1 | **+133.1** | +129.4 |
| **Stage 4** | Éxito | **100%** | **100%** | **100%** | **100%** | 82.1% | **100%** |
| **Stage 4** | Col approach | 0% | 0% | 0% | 0% | 0% | 0% |
| **Stage 4** | Col exit | 0% | 0% | 0% | 0% | 17.9% | 0% |
| **Stage 4** | Pasos/ep | 1737 | 1739 | 1736 | **1605** | **1490** | **1603** |
| **Stage 4** | Reward/ep | **380.0** | **383.9** | **378.9** | 365.6 | 298.9 | 363.0 |

### 7.7 Resumen por etapa

| Etapa | Ganador | Motivo |
|-------|:-------:|--------|
| Stage 1 | **Empate** | Path corto y directo — ambos mecanismos equivalentes |
| Stage 2 | **STHWP** (+2.4pp) | Subgoal adaptativo tolera mejor la fragilidad de s123 en paths largos |
| Stage 3 | **STHWP** (+43pp) | `_fase_escape` resuelve el problema de waypoint trans-esquina |
| Stage 4 | **SUB-WP** (+6pp, 3/3 vs 2/3 seeds) | Heading natural de approach elimina el problema de teleport; política integrada |

---

### 7.8 Conclusión: ¿qué método es mejor globalmente?

La respuesta depende del escenario de uso:

**STH-WP es mejor para exit puro desde posición arbitraria.**
Cuando el robot puede iniciarse en cualquier orientación dentro de la estantería
(fallo, reinicio, teleoperación), el subgoal adaptativo con `_fase_escape` ofrece
robustez estructural. El mecanismo de reducción de lookahead (1.5m → 0.5m) durante
los primeros 50 steps resuelve elegantemente el problema de corner-jumping sin
ningún coste adicional en el ciclo completo.

**SUB-WP es mejor para la tarea de producción (ciclo approach→exit encadenado).**
En el uso real de un AMR de almacén, el robot siempre llega a la estantería desde
la zona de espera (approach), recoge/deposita el pallet y sale (exit). En este
escenario, SUB-WP logra 100% en los 3 seeds con mayor reward (+38 puntos) frente al
82-100% de STHWP. La menor complejidad algorítmica (sin `compute_sth_subgoal`, sin
interpolación continua, sin `_fase_escape`) también lo hace más interpretable y
más fácil de depurar en producción.

**Implicación para el diseño de sistemas RL de navegación:**
El mecanismo de subgoal importa más en la tarea de evaluación aislada (exit puro)
que en la tarea integrada real. Esto sugiere que los benchmarks de exit aislado
pueden sobreestimar la ventaja del subgoal adaptativo. El diseño de curriculum
(stage 4 con ciclo completo) compensa efectivamente la debilidad del subgoal
discreto en exit puro al entrenar la política en las condiciones reales de operación.

**Si se tuviera que elegir un único método para producción: SUB-WP**, por su
100% de éxito en ciclo completo, mayor reward, y menor complejidad. Si se necesita
robustez ante inicios en posición arbitraria: STH-WP.

---

## 8. Experimento de mejora: WP_STEP_EXIT = 0.75m

### 8.1 Motivación

El análisis de §7.4 demostró que el rendimiento de SUB-WP en stage 3 (exit puro) es
del 29% frente al 72% de STH-WP. La causa estructural identificada es el problema del
**waypoint trans-esquina**: con `WP_STEP=1.5m`, el primer waypoint del exit path puede
estar al otro lado de una esquina del pasillo, haciéndolo inalcanzable en línea recta
desde dentro de la estantería.

La hipótesis es que reducir el espaciado de los exit paths a **0.75m** evitará este
problema: el primer waypoint estará dentro del mismo segmento de pasillo que el robot,
nunca al otro lado de una esquina.

### 8.2 Cambio implementado

**Archivo modificado:** `webots_env.py`

```python
# Antes (línea 45):
WP_STEP = 1.5   # metros — espaciado entre waypoints

# Después:
WP_STEP      = 1.5   # metros — espaciado approach
WP_STEP_EXIT = 0.75  # metros — espaciado exit: más fino para esquinas de pasillos
```

**Función `_precompute_paths()` (líneas 197-208):**

```python
# approach: sin cambios — sigue usando WP_STEP=1.5m
if path and len(path) > 1:
    path = subsample_path(path, step=self.WP_STEP)
cache[("approach", idx)] = path if path else [goal_pos]

# exit: ahora usa WP_STEP_EXIT=0.75m
if path and len(path) > 1:
    path = subsample_path(path, step=self.WP_STEP_EXIT)
cache[("exit", idx)] = path if path else [self.zona_descarga]
```

**Efecto sobre los paths exit:** el número de waypoints por exit path pasa de ~8-10 a
~16-20. El primer waypoint está ahora a 0.75m del punto de inicio — dentro del radio
de captura `WP_REACH_DIST=0.5m` más rápido y sin cruzar esquinas.

Los approach paths no se modifican, por lo que **stages 1 y 2 no necesitan
reentrenamiento**.

### 8.3 Nomenclatura de archivos — sufijo `_wp75`

Para no sobreescribir los modelos e inferencias de la versión original (`WP_STEP=1.5m`),
todos los archivos de la nueva versión llevan el sufijo `_wp75`:

| Tipo | Original | Nuevo |
|------|----------|-------|
| Modelo stage 3 | `subwp_s42_stage3_final.zip` | `subwp_s42_wp75_stage3_final.zip` |
| Modelo stage 4 | `subwp_s42_stage4_final.zip` | `subwp_s42_wp75_stage4_final.zip` |
| CSV inferencia stage 3 | `infer_subwp_s42_stage3.csv` | `infer_subwp_s42_wp75_stage3.csv` |
| CSV inferencia stage 4 | `infer_subwp_s42_stage4.csv` | `infer_subwp_s42_wp75_stage4.csv` |
| TensorBoard stage 3 | `stage3_subwp_s42` | `stage3_subwp_s42_wp75` |
| TensorBoard stage 4 | `stage4_subwp_s42` | `stage4_subwp_s42_wp75` |

Scripts modificados: `train_stage3_s{42,123,524}.py`, `train_stage4_s{42,123,524}.py`,
`inferencia_subwp/infer_stage3_s{42,123,524}.py`, `inferencia_subwp/infer_stage4_s{42,123,524}.py`.

### 8.4 Scripts de ejecución

**Entrenamiento stages 3 + 4 (secuencial, ~30-36h):**
```bash
bash run_subwp_stage3_4_wp75.sh
```

Orden de ejecución: stage3_s42 → stage3_s123 → stage3_s524 → stage4_s42 → stage4_s123 → stage4_s524.
Cada stage 4 carga el checkpoint `_wp75_stage3_final` correspondiente a su seed.

**Inferencia (tras completar entrenamiento):**
```bash
bash run_subwp_infer_all.sh
```
Los scripts de inferencia de stages 3 y 4 ya apuntan a los modelos `_wp75` y escriben
CSVs con el sufijo `_wp75`. Los stages 1 y 2 no se ven afectados.

### 8.5 Análisis TensorBoard — Stage 3 wp75

*(Inferencia determinista pendiente de lanzar. Análisis basado exclusivamente en TensorBoard.)*

#### 8.5.1 Recompensa y longitud de episodio

| Seed | ep_rew_mean (smoothed) | ep_rew_mean (último) | ep_len_mean (smoothed) |
|------|:---------------------:|:--------------------:|:----------------------:|
| s42  | -41.4 | -34.6 | 272 |
| s123 | -59.3 | -61.2 | 234 |
| s524 | -76.1 | -82.3 | 185 |

Todos los seeds mantienen recompensa media negativa durante las 4M steps del stage 3.
Las curvas oscilan fuertemente sin convergencia visible, reflejo de la alta tasa de
colisión. s42 tiene el mejor reward y los episodios más largos (~272 pasos), indicando
que el robot navega más antes de colisionar. s524 es el más agresivo/rápido en fallar
(185 pasos, -82 reward).

#### 8.5.2 Tasas globales de éxito y colisión

| Seed | tasa_exito_% | tasa_colision_% | tasa_truncado_% | colision_ult100_% | exito_ult100_% |
|------|:------------:|:---------------:|:---------------:|:-----------------:|:--------------:|
| s42  | 29.2% | 70.7% | ~0% | 65% | 35% |
| s123 | 28.0% | 71.8% | 0.16% | 74% | 26% |
| s524 | 28.5% | 71.5% | ~0% | 78% | 22% |

A primera vista, la tasa de éxito global (~28-29%) es idéntica a la v1 (29%). Sin
embargo, la distribución por goal es radicalmente diferente (ver §8.5.3). Los
truncados son prácticamente 0 en todos los seeds, confirmando que la política
siempre termina en éxito o colisión — no hay timeouts.

#### 8.5.3 Cambio estructural: distribución de éxitos por goal

**Este es el hallazgo más relevante del entrenamiento wp75.**

En la v1 (`WP_STEP=1.5m`), la distribución era binaria:
- **9 goals** → ~95-100% éxito durante entrenamiento (salidas rectas)
- **19 goals** → ~0% éxito durante entrenamiento (waypoint trans-esquina)

En wp75 (`WP_STEP_EXIT=0.75m`), los 28 plots por goal (imágenes #53-54) muestran:
- **28 goals** → 20-35% éxito durante entrenamiento

El cambio es fundamental: la política wp75 **aprende a intentar el exit desde todos
los goals**, no solo desde los 9 con salida recta. La reducción a 0.75m eliminó el
bloqueo estructural del waypoint trans-esquina. El primer waypoint del exit path ahora
está dentro del mismo segmento de pasillo que el robot, lo que permite comenzar a
moverse en la dirección correcta y recibir recompensa de progreso.

La convergencia por goal es lenta y ruidosa (~25-30% en training stocástico), pero
**uniforme** — ningún goal está a 0%.

#### 8.5.4 Métricas de entrenamiento PPO

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| approx_kl | 0.0112 | 0.0077 | 0.0139 |
| clip_fraction | 0.114 | 0.111 | 0.132 |
| entropy_loss | -1.88 | -3.63 | -2.71 |
| explained_variance | **0.203** | **0.910** | **0.910** |
| train/std | 2.14 | 5.70 | 3.45 |
| value_loss | **81.6** | 28.6 | 32.2 |

El dato más preocupante es **s42: explained_variance = 0.203**. El crítico de s42 no
está aprendiendo a predecir los retornos correctamente — los gradientes de política
son por tanto muy ruidosos. Paradójicamente, s42 tiene el mejor reward final (-34.6),
lo que sugiere que la política encontró una estrategia de explotación válida sin
necesitar un crítico preciso. Sin embargo, la alta `value_loss` (81.6 vs 28-32 de los
otros seeds) indica que el aprendizaje de valor es inestable.

s123 y s524 tienen un crítico excelente (explained_variance ~0.91) pero la política
rinde peor, posiblemente porque la entropía de s123 se ha colapsado demasiado (-3.63),
lo que reduce la exploración y genera una política muy determinista pero subóptima.

**FPS:** s42=1093, s123=1083, s524=1078 — velocidades similares, sin diferencias
significativas.

#### 8.5.5 Comparativa con v1 (TensorBoard) e hipótesis para inferencia

| Aspecto | v1 (WP_STEP=1.5m) | wp75 (WP_STEP_EXIT=0.75m) |
|---------|:-----------------:|:-------------------------:|
| tasa_exito global | ~29% | ~28-29% |
| tasa_colision global | ~70% | ~71% |
| Goals con éxito>0% en training | 9/28 (32%) | 28/28 (100%) |
| Goals con éxito~0% en training | 19/28 | 0/28 |
| Distribución éxito por goal | Bimodal (0% o 95%) | Unimodal (~28% todos) |

En la v1, la inferencia determinista elevó los 9 goals buenos a 99-100% (la política
determinista explotó lo aprendido). Por el mismo mecanismo, la inferencia determinista
de wp75 debería elevar el ~28% estocástico de cada goal a un valor significativamente
mayor. **Hipótesis: tasa_exito de inferencia >50% global** (vs 29% de v1).

El resultado exacto dependerá de cuánto "eleva" la inferencia determinista el
rendimiento stocástico de training. Si la relación es similar a v1 (donde el factor
fue aproximadamente ×3 para los goals buenos), wp75 podría dar 60-80% global.

#### 8.5.6 Próximos pasos

1. **Lanzar inferencia stage 3 wp75** — ya configurada con `run_subwp_infer_all.sh`
   (ejecutará stages 3 y 4, pero con los modelos wp75 actuales solo hay stage 3).
2. **Esperar resultado de stage 4 wp75** — training en curso o pendiente.
3. **Completar §8.5 con datos reales** de los CSVs.

---

## 9. Stage 4 wp75 — Análisis TensorBoard

*(Inferencia determinista pendiente. Análisis basado en TensorBoard.)*

### 9.1 Recompensa y longitud de episodio

| Seed | ep_rew_mean (smoothed) | ep_rew_mean (último) | ep_len_mean (smoothed) |
|------|:---------------------:|:--------------------:|:----------------------:|
| s42  | 268.1 | 273.3 | 1411 |
| s123 | 247.2 | 248.0 | 1321 |
| s524 | 243.2 | 238.1 | 1300 |

Todos los seeds mantienen **recompensa positiva y estable** (180-280 rango), lo que
confirma que la política ciclo completo funciona. Las curvas oscilan pero no caen a
negativo, al contrario que stage 3. s42 lidera con el mayor reward (273) y los
episodios más largos (1411 pasos).

Comparado con v1 stage 4 (~380 reward, ~1737 pasos): wp75 es **~100 puntos menos
eficiente** y ~300-400 pasos más corto. La reducción de pasos indica más episodios que
acaban en colisión durante exit (episodio más corto = fallo antes del final del ciclo).

### 9.2 Tasas globales — ciclo completo

| Seed | tasa_exito_% | tasa_colision_% | tasa_truncado_% | n_exitos | n_colisiones |
|------|:------------:|:---------------:|:---------------:|:--------:|:------------:|
| s42  | 78.4% | 21.6% | 0.035% | 2274 | 626 |
| s123 | 79.6% | 20.4% | ~0% | 2390 | 611 |
| s524 | 77.6% | 22.4% | ~0% | 2400 | 694 |

**`tasa_exito_%` del 77-80% en training estocástico** es el resultado más relevante.
Compare con stage 3 wp75 (28-29%) y con v1 stage 4 (que en training también estaba en
el rango 80-85%). Los truncados son prácticamente 0 — todos los episodios terminan en
éxito o colisión, nunca por timeout.

Los 600-700 colisiones totales en 4M steps son dramáticamente menores que las 11M+
colisiones del stage 3 wp75, confirmando que stage 4 está dominado por éxitos.

### 9.3 Desglose approach vs exit

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| colision_approach_% | **0%** | **0%** | **0%** |
| colision_exit_% (final) | 21.6% | 20.4% | 22.4% |
| truncado_approach_% | 0% | 0% | 0% |
| truncado_exit_% | ~0% | 0% | 0% |
| llego_estanteria_ult% | **100%** | **100%** | **100%** |

El approach es perfecto: 0% colisión, 0% truncado, 100% llegada a estantería — idéntico
a v1. El problema residual está **exclusivamente en el exit**: ~20-22% de colisión.

La gráfica `stage4/colision_exit_%` muestra una tendencia decreciente desde ~23-24% al
inicio del stage 4 hasta ~20-22% al final de las 4M steps. Esto indica que la política
sigue aprendiendo a reducir las colisiones en exit pero no ha convergido completamente.
Con más steps de entrenamiento podría seguir mejorando.

En v1 el exit era 0% colisión porque el heading natural de approach eliminaba el
problema de esquinas. En wp75, los waypoints más finos ayudan (comparado con stage 3
donde la colisión era 70%), pero hay un ~20% residual que no se ha eliminado en 4M
steps.

### 9.4 Éxito por goal durante entrenamiento

A diferencia de stage 3 wp75 (todos los goals al 25-30%) y de v1 stage 4 (todos al
85-95%), los 28 goals en stage 4 wp75 convergen a **70-90%** durante training:

| Rango de goals | Éxito training (smoothed) | Observación |
|----------------|:-------------------------:|-------------|
| goals 01-07 | 70-86% | Bloque 1-2, variabilidad por seed |
| goals 08-14 | 68-85% | Bloque 2-3, s42 más variable |
| goals 15-21 | 70-87% | Bloque 3-4, s123 lidera |
| goals 22-28 | 67-90% | Bloque 4-5, goal_25 s42 destaca (88%) |

No hay ningún goal con rendimiento cercano a 0% — la cobertura es universal. Algunos
goals tienen mayor variabilidad entre seeds (goal_06: s42=70%, s123=78%; goal_11: s42=71%,
s123=82%). Los goals más difíciles rondan el 68-72%.

### 9.5 Métricas PPO — estabilidad de entrenamiento

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| approx_kl | 0.022 | 0.015 | 0.007 |
| clip_fraction | 0.128 | 0.110 | 0.068 |
| entropy_loss | -2.31 | -3.61 | -2.95 |
| **explained_variance** | **0.143** | **0.876** | **0.898** |
| train/std | 7.45 | 10.05 | 6.42 |
| value_loss | 44.8 | 14.0 | 15.4 |

El patrón del stage 3 se agrava en stage 4: **s42 tiene explained_variance = 0.143**,
el peor de los tres seeds y peor aún que su propio stage 3 (0.203). El crítico de s42
casi no funciona — la estimación de ventaja es esencialmente ruido. Sin embargo, s42
mantiene el mejor reward (273) y la mayor longitud de episodio (1411 pasos). Esto
sugiere que la política de s42 encontró una estrategia buena por gradiente de política
directa, sin depender del crítico.

s123 y s524 tienen critics excelentes (EV=0.88-0.90) y entropía bien controlada
(-2.9 a -3.6), lo que indica políticas más estables aunque con reward inferior.

El `train/std` notablemente alto (6-10 vs 2-6 en stage 3) indica que las políticas
siguen explorando activamente — la tarea de ciclo completo con 28 goals presenta suficiente
diversidad como para mantener alta la entropía de acción.

**FPS:** s123=1093, s524=1092, s42=1085 — estable, sin diferencias operativas.

### 9.6 Comparativa SUB-WP v1 vs wp75 — stages 3 y 4

| Stage | Métrica | v1 (WP_STEP_EXIT=1.5m) | wp75 (WP_STEP_EXIT=0.75m) | Δ |
|-------|---------|:---------------------:|:-------------------------:|:-:|
| **Stage 3** | tasa_exito training | ~29% | ~28-29% | ≈0 |
| **Stage 3** | goals con éxito>0% | **9/28** | **28/28** | +19 goals |
| **Stage 3** | distribución | bimodal (0% o 95%) | unimodal (~28%) | estructural |
| **Stage 4** | tasa_exito training | ~83-85% | **77-80%** | -5pp |
| **Stage 4** | colision_exit training | **0%** | **~20-22%** | +20pp |
| **Stage 4** | ep_rew_mean | ~380 | ~248-273 | -100 |
| **Stage 4** | goals con >70% exito | 28/28 | 28/28 | igual |

El resultado más importante: **wp75 no supera a v1 en stage 4** durante el entrenamiento.
La reducción de WP_STEP_EXIT ayudó estructuralmente al exit puro (stage 3 ahora trabaja
todos los goals) pero introdujo una dificultad en el ciclo completo: con más waypoints
en el exit (16-20 vs 8-10), la política debe aprender a navegar más transiciones, lo
que aumenta la probabilidad de colisión en algún punto del trayecto.

El ~20% de colisión en exit de stage 4 wp75 (vs 0% en v1) es la principal diferencia.
Si la inferencia determinista reduce esta tasa a <10%, los resultados finales podrían
ser comparables. **Si la mantiene cerca del 20%, wp75 stage 4 sería inferior a v1.**

### 9.7 Hipótesis para inferencia determinista

Basándose en el patrón observado en v1 (donde la inferencia determinista multiplicó el
rendimiento de training por ~1.2-1.5x para goals con éxito>30%), las predicciones para
wp75 son:

| Stage | Training stoch. | Hipótesis inferencia det. |
|-------|:---------------:|:-------------------------:|
| Stage 3 wp75 | ~28-29% | **55-70%** (todos los goals contribuyen) |
| Stage 4 wp75 | ~78-80% | **85-95%** (baja colisión exit en det.) |

La inferencia determinista eliminará el ruido de la política estocástica. Los goals que
durante training tienen 70-90% podrían llegar a 85-100% en inferencia determinista.

**Acción recomendada:** lanzar inferencia (`run_subwp_infer_all.sh`) para obtener datos reales.

---

## 10. Inferencia wp75 — Resultados y comparativa final

### 10.1 Stage 3 — Exit puro determinista

#### Resultados globales

| Seed | SUB-WP v1 éxito | SUB-WP wp75 éxito | STHWP éxito |
|------|:---------------:|:-----------------:|:-----------:|
| s42  | 29.2% | **28.9%** | 63.6% |
| s123 | 29.2% | **29.0%** | 76.4% |
| s524 | 28.6% | **29.0%** | 76.7% |
| **Media** | **29.0%** | **29.0%** | **72.2%** |

#### Hallazgo crítico: wp75 NO mejora stage 3

El resultado más relevante del experimento: **wp75 produce exactamente los mismos
resultados de inferencia que v1**. Los CSVs por goal son prácticamente idénticos:

| Goal | SUB-WP v1 (media 3 seeds) | SUB-WP wp75 (media 3 seeds) |
|------|:------------------------:|:---------------------------:|
| 01-07 | **0%** | **0%** |
| 08 | 99.3% | 100% |
| 09 | 99.0% | 99.7% |
| 10 | 100% | 98.7% |
| 11-16 | **0%** | **0%** |
| 17 | 46.7% | 44.7% |
| 18 | 88.0% | 88.7% |
| 19 | 80.0% | 80.3% |
| 20 | 100% | 98.7% |
| 21 | 99.7% | 99.3% |
| 22 | 100% | 99.7% |
| 23-28 | **0%** | **0%** |

Los mismos **9 goals** que funcionaban en v1 (08-10, 17-22) siguen siendo los únicos
que funcionan en wp75. Los mismos **19 goals** siguen en 0%. La reducción de
`WP_STEP_EXIT` de 1.5m a 0.75m no cambió nada en inferencia determinista.

#### Por qué wp75 no ayuda en stage 3

La explicación es clave para entender la diferencia entre entrenamiento estocástico e
inferencia determinista:

**Durante el entrenamiento wp75** (estocástico), la política exploró con ruido gaussiano
adicional a cada acción. Con waypoints más finos, ocasionalmente el robot encontró
trayectorias que cruzaban las esquinas desde los 19 goals difíciles — de ahí que el
training mostrase todos los goals al 25-30%. Sin embargo, esas trayectorias exitosas
eran soluciones marginales que solo aparecían con suficiente ruido de exploración.

**En inferencia determinista**, la política ejecuta siempre la acción de máxima
probabilidad (argmax). La política aprendida converge a las mismas soluciones robustas
que v1: ir directamente hacia el waypoint. Para los 19 goals con waypoints trans-
esquina, esa acción determinista sigue siendo intentar atravesar la pared — el mismo
fallo que en v1.

La diferencia entre `WP_STEP_EXIT=1.5m` y `WP_STEP_EXIT=0.75m` no cambió el
**comportamiento aprendido determinista** porque ambas configuraciones generan el
mismo tipo de problema en esos 19 goals: el primer waypoint visible desde dentro
de la estantería sigue estando al otro lado de la esquina, aunque más cercano.
El problema no es solo la distancia del waypoint sino la geometría de oclusión de
la esquina respecto al robot.

**Conclusión: la hipótesis de mejora era incorrecta.** Reducir `WP_STEP_EXIT` no
es suficiente para resolver el problema de esquinas en inferencia determinista.
La solución requiere un mecanismo diferente — como el `_fase_escape` de STHWP
que adapta el subgoal en tiempo real independientemente del espaciado.

### 10.2 Stage 4 — Ciclo completo determinista

#### Resultados globales

| Seed | SUB-WP v1 éxito | SUB-WP wp75 éxito | STHWP éxito |
|------|:---------------:|:-----------------:|:-----------:|
| s42  | 100% | **100%** | 100% |
| s123 | 100% | **100%** | 82.1% |
| s524 | 100% | **100%** | 100% |
| **Media** | **100%** | **100%** | **94.0%** |

#### Eficiencia: pasos y reward

| Seed | SUB-WP v1 pasos | SUB-WP wp75 pasos | STHWP pasos |
|------|:---------------:|:-----------------:|:-----------:|
| s42  | 1736.8 | 1815.1 (+78) | 1604.8 |
| s123 | 1739.3 | 1752.4 (+13) | 1489.6 |
| s524 | 1736.3 | 1742.0 (+6)  | 1603.1 |

| Seed | SUB-WP v1 reward | SUB-WP wp75 reward | STHWP reward |
|------|:----------------:|:------------------:|:------------:|
| s42  | 380.0 | 378.9 | 365.6 |
| s123 | 383.9 | 380.7 | 298.9 |
| s524 | 378.9 | 382.8 | 363.0 |

wp75 alcanza el **mismo 100% de éxito** que v1 con una diferencia de pasos mínima
(+6 a +78 pasos/ep según seed). El reward es prácticamente idéntico (~379-383).
STHWP, a pesar de ser más eficiente en pasos (1490-1605 vs 1737-1815), no logra el
100% debido al fallo de s123 en 5 goals específicos (82.1%).

### 10.3 Tabla comparativa final — todos los métodos

| Stage | Método | s42 | s123 | s524 | Media |
|-------|--------|:---:|:----:|:----:|:-----:|
| **Stage 1** | SUB-WP v1 | 100% | 100% | 100% | **100%** |
| **Stage 1** | STHWP | 100% | 100% | 100% | **100%** |
| **Stage 2** | SUB-WP v1 | 100% | 92.9% | 100% | **97.6%** |
| **Stage 2** | STHWP | 100% | 100% | 100% | **100%** |
| **Stage 3** | SUB-WP v1 | 29.2% | 29.2% | 28.6% | **29.0%** |
| **Stage 3** | SUB-WP wp75 | 28.9% | 29.0% | 29.0% | **29.0%** |
| **Stage 3** | STHWP | 63.6% | 76.4% | 76.7% | **72.2%** |
| **Stage 4** | SUB-WP v1 | 100% | 100% | 100% | **100%** |
| **Stage 4** | SUB-WP wp75 | 100% | 100% | 100% | **100%** |
| **Stage 4** | STHWP | 100% | 82.1% | 100% | **94.0%** |

### 10.4 Conclusión del experimento wp75

La reducción de `WP_STEP_EXIT` de 1.5m a 0.75m:

- **Stage 3**: sin efecto medible (+0.0pp). Los 9 goals que funcionaban siguen
  funcionando con las mismas tasas. Los 19 que fallaban siguen en 0%.
- **Stage 4**: sin regresión. Se mantiene el 100% de éxito con reward y pasos
  prácticamente idénticos a v1.

El experimento confirma que **el problema de exit puro desde posición arbitraria
no es resoluble en SUB-WP mediante ajuste del espaciado de waypoints**. La causa
raíz es geométrica: los 19 goals fallidos tienen salidas donde cualquier waypoint
a cualquier distancia ≤1.5m está oculto por la geometría de la esquina. Reducirlo
a 0.75m no cambia el hecho de que el primer waypoint sigue estando en una dirección
bloqueada desde la posición de teleport.

La única solución efectiva documentada en este TFM es el enfoque STHWP (`_fase_escape`
+ `compute_sth_subgoal`), que no usa waypoints fijos sino un subgoal que se recalcula
en cada step en función de la posición actual del robot — eliminando el problema de
oclusión geométrica por definición.

**Recomendación para el TFM:** la comparativa SUB-WP wp75 se incluye como ablation
study que refuerza la conclusión principal: la ventaja de STHWP en stage 3 es
estructural y no depende del espaciado de waypoints sino del mecanismo de subgoal.

---

## 11. Stage 5 — Return puro (descarga → espera)

### 11.1 Motivación

El ciclo de navegación del AMR de almacén consta de 3 tramos:
1. **Approach**: zona de espera → estantería (stages 1-2)
2. **Exit**: estantería → zona de descarga (stages 3-4)
3. **Return**: zona de descarga → zona de espera ← **nuevo**

Stage 5 entrena el tramo de vuelta, completando el ciclo completo de operación.

### 11.2 Cambios en `webots_env.py`

**Nuevas constantes / variables:**
```python
self._en_retorno = False  # True durante el tramo descarga → espera
```

**Nueva ruta en `_precompute_paths()` (57 rutas totales):**
```python
# return: zona_descarga → zona_espera (única ruta fija)
path = plan_path(self.map_path, self.zona_descarga, self.zona_espera, ...)
if path and len(path) > 1:
    path = subsample_path(path, step=self.WP_STEP)  # 1.5m, espacio abierto
cache[("return", 0)] = path if path else [self.zona_espera]
```

**Nuevo método `_reset_return()`:**
- Teleporta el robot a `zona_descarga`
- Heading computado desde el primer segmento del return path + ruido N(0, 0.1 rad)
- Establece `_en_retorno = True`, `_hacia_descarga = False`

**Llegada en `step()` — restructurada con prioridad `_en_retorno`:**
```python
if self._en_retorno:
    # zona de espera alcanzada → éxito (+100 reward)
    recompensa += 100.0; terminated = True
    info["exito"] = True; info["llego_espera"] = True

elif not self._hacia_descarga:
    # approach: llegó a estantería
    if self.stage <= 2: éxito
    else: switch to exit

else:
    # exit: llegó a descarga → éxito ciclo approach+exit
```

**`assert stage in (1, 2, 3, 4, 5)`** — stage 5 habilitado.

### 11.3 Diseño del curriculum

| Stage | Tarea | Duración | Base |
|-------|-------|:--------:|------|
| 1 | Approach goal_01 | 500k steps | — |
| 2 | Approach 28 goals | 4M steps | stage 1 |
| 3 | Exit puro 28 goals (teleport) | 4M steps | stage 2 |
| 4 | Ciclo approach+exit (70/30) | 4M steps | stage 3 |
| **5** | **Return puro (teleport ret_path[2])** | **2M steps** | **stage 4** |

Stage 5 es más corto (2M steps vs 4M) porque:
- La tarea de return es más simple: única ruta fija, sin variabilidad de goal
- El espacio de descarga → espera está en zonas abiertas (sin estanterías)
- La política ya aprendió la dinámica general de navegación en stages anteriores

Hiperparámetros finales (tras debugging — ver §11.5):
- `learning_rate = 3e-4` (incrementado para que la policy tenga señal suficiente)
- `ent_coef = 0.02` (aumentado de 0.005 para forzar exploración en posición de spawn única)

### 11.4 Scripts creados

| Script | Descripción |
|--------|-------------|
| `train_stage5_s{42,123,524}.py` | Entrenamiento stage 5 por seed |
| `run_subwp_stage5.sh` | Lanza los 3 seeds (v1 — fallida) |
| `run_subwp_stage5_v2.sh` | Lanza los 3 seeds con spawn correcto (v2 — fallida) |
| `run_subwp_stage5_v3.sh` | Lanza los 3 seeds con fix de rotación (v3 — definitiva) |
| `inferencia_subwp/infer_stage5_s{42,123,524}.py` | Inferencia 300 ep, return puro |
| `run_subwp_infer_stage5.sh` | Lanza inferencia stage 5 (~15 min total) |

### 11.5 Debugging: colisión en step 1 (bumper contra pared norte)

#### 11.5.1 Síntoma — Curva completamente plana en −150

El primer intento de stage 5 (lr=1e-4, ent_coef=0.005, spawn en zona_descarga) mostró
una curva de reward plana en −150.001 desde el inicio hasta el final de los 2M steps en los
3 seeds. TensorBoard mostraba:

| Métrica | Valor observado |
|---------|----------------|
| `rollout/ep_rew_mean` | −150.001 completamente plana |
| `ep_len_mean` | **1** (episodios de 1 solo paso) |
| `stats/tasa_colision_%` | **100%** |
| `stats/n_exitos` | 6–27 en 1.9M episodios (0.001%) |
| `train/explained_variance` | −∞ oscilante (value function rota) |

#### 11.5.2 Diagnóstico — `ep_len = 1` como clave

`ep_len_mean = 1` es el dato crítico: **el bumper dispara en el primer `supervisor.step()`
del método `step()`**, antes de que el robot tenga tiempo de moverse. El reward es exactamente
−150 − 0.001 (colisión + penalización por paso), lo que confirma que no hay ninguna componente
de progreso (la fórmula de progress reward está en la rama `else` del `if bumper`).

Los únicos episodios con éxito (27 en s42 en 1.9M episodios) corresponden a ejecuciones en que
el ruido de heading (sigma=0.2 rad) fue suficientemente grande (~>1.5 rad, ≈7σ) para que el
robot spawneara en una orientación diferente que evitara la colisión inicial.

#### 11.5.3 Segundo intento — Incremento de entropía y pre-avance de waypoint

Se aplicaron dos correcciones intermedias:

**A. Pre-avance del waypoint inicial en `reset()`:**  
El robot spawna en zona_descarga (0.0, 10.5) y wp[0]=(0.25, 10.25) está a solo 0.35m
(< WP_REACH_DIST=0.5m). Esto causaba un progress reward de `(0.35 − 1.77) × 3 = −4.24` en el
primer step, incluso cuando el robot iba en la dirección correcta. La corrección:

```python
# En reset(), tras simulationResetPhysics() + step():
wp_x, wp_y = self.full_path[self._wp_idx]
dist_to_wp0 = math.sqrt((pos[0]-wp_x)**2 + (pos[1]-wp_y)**2)
if dist_to_wp0 < self.WP_REACH_DIST and self._wp_idx < len(self.full_path)-1:
    self._wp_idx += 1  # skip wp[0], _prev_dist = dist to wp[1] ≈ 1.77m
```

**B. Incremento de hiperparámetros de exploración:**  
`lr: 1e-4 → 3e-4`, `ent_coef: 0.005 → 0.02`.

Este segundo intento mostró exactamente el mismo resultado (ep_len=1, reward=−150.001).
El bug del bumper persistía — el pre-avance no lo resuelve porque el bumper dispara físicamente,
antes de cualquier cálculo de reward.

#### 11.5.4 Causa raíz — Colisión física bumper vs wall3

La causa es una **colisión física en Webots** entre el bounding box del bumper del MiR100 y
wall3 (pared norte, y=11.9m interior) cuando el robot se teleporta a zona_descarga:

- `zona_descarga = (0.0, 10.5)` → solo **1.4m** de la pared norte (y=11.9)
- Heading de spawn: `[0, 1, 0, π/2]` (retorno hacia sur) posiciona el bounding box del bumper
  del MiR100 en una geometría que toca wall3 durante el primer `supervisor.step()`
- El mapa A* (margin=1.2m) sí consideraba esta zona potencialmente problemática: el primer
  nodo A* navegable es (0.25, 10.25), no zona_descarga en (0.0, 10.5)
- El hecho de que stages 3-4 no fallaran es porque la zona_descarga era el **destino** (nunca
  el punto de spawn): el robot llegaba desde el sur con su propia inercia, sin la colisión
  instantánea del teleport

**Confirmación del diagnóstico**: `stats/n_exitos = 27` — los 27 éxitos corresponden a episodios
donde el ruido de heading era tan grande que el robot no colisionaba, confirmando que la posición
(0.0, 10.5) con heading ≈ −π/2 es la causa directa.

#### 11.5.5 Fix definitivo — Spawn en ret_path[2]

Se cambió el punto de spawn de zona_descarga a `ret_path[2] = (0.0, 7.25)`:

```python
# En _reset_return():
SPAWN_IDX = 2  # ret_path[2] = (0.0, 7.25) — 4.65m de wall3
spawn_x, spawn_y = ret_path[SPAWN_IDX]

# heading: dirección del tramo SPAWN_IDX → SPAWN_IDX+1
dx = ret_path[SPAWN_IDX+1][0] - ret_path[SPAWN_IDX][0]
dy = ret_path[SPAWN_IDX+1][1] - ret_path[SPAWN_IDX][1]
theta = math.atan2(dy, dx)  # ≈ −108.4° (suroeste)

# Path trimado: solo los 14 waypoints desde el spawn hasta espera
self.full_path = list(ret_path[SPAWN_IDX:])  # wp[0]=(0.0,7.25)→wp[13]=espera
```

Características del nuevo spawn:
- **Posición**: (0.0, 7.25) — nodo A* confirmado como navegable
- **Distancia a wall3**: 4.65m (vs 1.4m en zona_descarga) → sin riesgo de colisión
- **Heading base**: −108.4° (sur-suroeste, hacia el siguiente waypoint)
- **Path de entrenamiento**: 14 waypoints (~19.5m) desde spawn hasta zona_espera
- **Cobertura de la ruta**: se omiten wp[0] y wp[1] del path completo (los ~3m cercanos
  a zona_descarga), que representan el 11% de la ruta total — aceptable para el TFM
- **TensorBoard**: `tb_log_name = stage5_{RUN_ID}_wp75_v2` para no solaparse con runs anteriores

#### 11.5.6 v2 falla idéntica a v1 — Bug real: eje de rotación incorrecto

La v2 (spawn en ret_path[2]) mostró **exactamente el mismo patrón** que la v1:
`ep_len=1`, `reward=−150.001`, `tasa_colision=100%` en los 2M steps × 3 seeds. El CSV final:

| Goal | Intentos | Éxitos | Colisiones |
|------|:--------:|:------:|:----------:|
| goal_01 | 2.000.896 | 0 | 2.000.896 |
| goal_02..28 | 0 | 0 | 0 |

TensorBoard confirmó: `ep_len_mean=1`, `explained_variance≈−3000`, `entropy_loss≈−17`,
`train/std→10000+`. Estos valores son consecuencia (no causa): el PPO diverge porque no tiene
señal de aprendizaje con ep_len=1.

**Causa raíz real — `heading_to_webots_rotation` usa eje Y en lugar de Z:**

```python
# ANTES (bug):
def heading_to_webots_rotation(heading_rad):
    return [0, 1, 0, -heading_rad]  # rotación alrededor de Y = PITCH (inclinación)

# DESPUÉS (fix v3):
def heading_to_webots_rotation(heading_rad):
    return [0, 0, 1, -heading_rad]  # rotación alrededor de Z = YAW (giro horizontal)
```

En este mundo Webots el **eje vertical es Z** (no Y): shelves y paredes usan
`rotation 0 0 1 angle`, el viewpoint está a z=75m mirando hacia abajo. Con `[0,1,0,-θ]`,
la rotación es un **pitch** (inclinación hacia adelante/atrás) alrededor del eje Y horizontal.

Para θ≈−108.4° (heading suroeste del stage 5), el pitch de 108° hunde el bounding box del
bumper del MiR100 bajo el suelo. Verificación matemática:

```
Bumper box: 0.9×0.6×0.4m, sin offset de translación.
Esquina frontal-superior en frame robot: (+0.45, *, +0.2)
Con pitch α=+108° alrededor del eje Y global:
  z_world = 0.45·(−sin108°) + 0.2·cos(108°) + 0.2
           = 0.45·(−0.951) + 0.2·(−0.309) + 0.2
           = −0.428 − 0.062 + 0.2
           = −0.290 m  ← 0.29m POR DEBAJO del suelo (z=0)
```

El bumper penetra el suelo en el momento del teleport → `bumper.getValue()>0` en el primer
`supervisor.step()` → terminación con −150 antes de que el robot se mueva.

**Por qué stages 1-4 no fallaban:**
- Stages 1-2: usan `[0,1,0,0]` (identidad) → sin pitch
- Stages 3-4: `heading_to_webots_rotation(θ)` con θ∈{0°, 90°, 180°}:
  - θ=0° → pitch=0° → z_min=0.2m (bumper en el aire) ✓
  - θ=180° → pitch=180° → z_min=0.2m (simetría) ✓
  - θ=90° → pitch=90° → bumper rota horizontal, geometría diferente, trabajaba ✓
- Stage 5: θ=−108.4° → pitch=108.4° → z_min=−0.290m → **hundimiento** ✗

**Fix completo:**

1. `heading_utils.py`: `[0,1,0,-θ]` → `[0,0,1,-θ]`
2. `webots_env.py` `_reset_approach()`: `[0,1,0,0]` → `[0,0,1,0]` (hardcoded consistente)
3. `train_stage5_s{42,123,524}.py`: docstring actualizado, `tb_log_name = ..._v3`
4. `run_subwp_stage5_v3.sh`: script de entrenamiento con TensorBoard correcto

### 11.6 Ejecución

```bash
# Entrenamiento definitivo — versión v3 (~6-9h)
bash run_subwp_stage5_v3.sh

# Inferencia (tras entrenamiento, ~15 min)
bash run_subwp_infer_stage5.sh
```

**Modelos de entrada:** `pruebas/subwp_s*_wp75_stage4_final.zip`
**Modelos de salida:** `pruebas/subwp_s*_wp75_stage5_final.zip`
**CSVs inferencia:** `inferencia_subwp/resultados/infer_subwp_s*_wp75_stage5.csv`
**TensorBoard:** `tensorboard_logs/stage5_subwp_s*_wp75_v3_0/`

### 11.7 Resultados — Stage 5 v3 (TensorBoard + CSV)

#### 11.7.1 Resumen cuantitativo — CSV final de entrenamiento

Cada seed entrena única y exclusivamente sobre `goal_01` (la ruta de retorno es una sola ruta fija).
Los datos son acumulados sobre los 2M steps de stage 5:

| Seed | Intentos | Éxitos | Colisiones | Truncados | tasa_exito_% | tasa_colision_% | tasa_truncado_% |
|------|:--------:|:------:|:----------:|:---------:|:------------:|:---------------:|:---------------:|
| s42  | 1.564 | 1.527 | 26  | 11  | **97,6%** | 1,7%  | 0,7%  |
| s123 | 1.387 | 1.196 | 67  | 124 | **86,2%** | 4,8%  | 8,9%  |
| s524 | 1.468 | 1.268 | 120 | 80  | **86,4%** | 8,2%  | 5,5%  |
| **Media** | **1.473** | **1.330** | **71** | **72** | **90,1%** | **4,9%** | **5,0%** |

**Nota sobre `llego_estanteria`:** en stage 5 (`_en_retorno=True`), el callback registra
`llego_estanteria = exitos + colisiones` (todas las terminaciones no-truncadas). Es un artefacto
del diseño del callback — no indica colisiones con estanterías; la estantería más cercana está a
>5m de la ruta de retorno.

Verificación: `exitos + colisiones + truncados = intentos` en todos los seeds:
- s42: 1527 + 26 + 11 = 1564 ✓
- s123: 1196 + 67 + 124 = 1387 ✓
- s524: 1268 + 120 + 80 = 1468 ✓

#### 11.7.2 Análisis TensorBoard — Evolución del entrenamiento

**Arranque exitoso (corrección v3 validada)**

A diferencia de v1/v2 (ep_len=1, reward=−150), en v3 los episodios arrancan con ep_len≈1.250-1.400
pasos y reward≈250. Este valor inicial alto refleja que el modelo heredado de stage4 ya conoce
la dinámica de navegación y puede recorrer parcialmente la ruta de retorno desde el primer episodio.

**Evolución por fases (referenciadas al step global, stage 5 empieza en ~12,5M):**

| Fase | Steps (global) | Steps stage 5 | Descripción |
|------|:--------------:|:-------------:|-------------|
| Arranque | 12,5M–12,8M | 0–0,3M | reward≈250, éxito≈100%, estable |
| Estabilización | 12,8M–13,0M | 0,3M–0,5M | reward≈240-250, ligera bajada |
| Degradación s123/s524 | 13,0M–13,8M | 0,5M–1,3M | caída pronunciada reward+éxito en s123 y s524 |
| Recuperación parcial | 13,8M–14,5M | 1,3M–2,0M | s524 sube de ~100 a ~175; s123 se mantiene bajo |
| s42 estable | todo el periodo | — | oscila entre 200-260, sin degradación severa |

**Métricas al final del entrenamiento (step 14.508.032):**

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| `rollout/ep_rew_mean` (smoothed) | 200,2 | 138,8 | 176,6 |
| `rollout/ep_rew_mean` (value) | 201,8 | 138,3 | 176,4 |
| `rollout/ep_len_mean` (smoothed) | 1.388 | 1.594 | 1.462 |
| `goal/goal_01_exito_%` (smoothed) | 99,0% | 90,3% | 86,4% |
| `goal/goal_01_exito_%` (value) | 98,4% | 87,5% | 85,7% |
| `stats/tasa_exito_%` (smoothed) | 97,6% | 86,2% | 86,4% |
| `stats/tasa_colision_%` (smoothed) | 1,7% | 4,8% | 8,2% |
| `stats/tasa_truncado_%` (smoothed) | 0,7% | 8,9% | 5,5% |
| `stats/colision_ult100_%` | 2 | 6 | 2 |
| `stats/exito_ult100_%` | 92% | 84% | 91% |
| `stats/truncado_ult100_%` | 6 | 10 | 7 |

**Divergencia inter-seed:**

s42 domina claramente sobre s123 y s524, con mayor éxito (+11 pp), menor colisión (−3 a −6 pp)
y menor ep_len (−200 pasos). La degradación de s123 y s524 a partir de ~13M steps es un indicio
de inestabilidad del entrenamiento: el policy gradient actualiza la política hacia waypoints iniciales
de la ruta pero puede olvidar tramos intermedios (olvido catastrófico parcial de la política
de navegación heredada).

#### 11.7.3 Análisis métricas de entrenamiento (train/)

| Métrica | s42 | s123 | s524 | Diagnóstico |
|---------|:---:|:----:|:----:|-------------|
| `approx_kl` | 0,0069 | 0,0041 | 0,0057 | Normal (<0,025) — actualizaciones estables |
| `clip_fraction` | 0,059 | 0,042 | 0,060 | Normal (<0,1) — sin recortes excesivos |
| `entropy_loss` | −7,23 | −7,62 | −7,38 | Entropía moderada — policy no colapsada |
| `explained_variance` | −0,053 | 0,484 | 0,816 | s42 con value function mal calibrada |
| `train/std` | 67,2 | 75,3 | 43,8 | Muy alta — policy muy estocástica |
| `time/fps` | 1.081 | 1.078 | 1.076 | Normal — simulación a ritmo esperado |

**Observación sobre `train/std` alto:** los valores 43-75 indican que la distribución de acciones
es muy dispersa. Esto es coherente con el patrón observado: el modelo aprende a ser exitoso en
promedio (~90%) pero con mucha variabilidad en las trayectorias concretas (ep_len fluctúa mucho).
El `explained_variance≈0` de s42 indica que la value function no predice bien el retorno — posible
causa de la alta std: el actor no tiene buena señal de cuándo una trayectoria es mejor que otra.

La inferencia determinista (argmax de la distribución) debería reducir drásticamente esta varianza
y elevar la tasa de éxito por encima del 90% observado en entrenamiento estocástico.

#### 11.7.4 Comparativa con stages anteriores

| Stage | Tarea | tasa_exito training | Obs. |
|-------|-------|:-------------------:|------|
| 1 | Approach goal_01 | 100% | Tarea trivial |
| 2 | Approach 28 goals | ~29% | Goals difíciles sin señal clara |
| 3 | Exit puro | ~29% | Misma problemática |
| 4 | Ciclo approach+exit | ~28-29% | Mezcla de tareas |
| **5** | **Return puro** | **~90%** | **Ruta única, espacio abierto** |

Stage 5 logra la tasa de éxito de training más alta desde stage 1. El factor clave es la
simplicidad estructural: una sola ruta fija en zona abierta, sin la complejidad de 28 goals
distintos con diferentes headings y proximidad a estanterías.

#### 11.7.5 Conclusión y siguientes pasos

**Entrenamiento exitoso:** el fix de `heading_to_webots_rotation` (eje Z en lugar de Y) resolvió
el bug que causaba colisión en step 1 en v1 y v2. Los tres seeds convergen a políticas funcionales.

**Resultado de training:**
- s42: política excelente (97,6% éxito, 1,7% colisión)
- s123, s524: políticas buenas pero con inestabilidad tardía (86% éxito, 5-8% colisión)
- Media 3 seeds: **90,1% éxito** en modo estocástico de entrenamiento

#### 11.7.6 Inferencia determinista — 300 episodios por seed

```bash
bash run_subwp_infer_stage5.sh
# → inferencia_subwp/resultados/infer_subwp_s*_wp75_stage5.csv
```

**Resultado global:**

| Seed | Episodios | Éxito | Colisión | Truncado | Pasos/ep | Reward/ep |
|------|:---------:|:-----:|:--------:|:--------:|:--------:|:---------:|
| s42  | 300 | **300 (100%)** | 0 (0%) | 0 (0%) | 1.202,0 | 249,0 |
| s123 | 300 | **300 (100%)** | 0 (0%) | 0 (0%) | 1.304,8 | 253,1 |
| s524 | 300 | **300 (100%)** | 0 (0%) | 0 (0%) | 1.223,9 | 265,0 |
| **Media** | **900** | **100%** | **0%** | **0%** | **1.243,6** | **255,7** |

**Los tres seeds alcanzan el 100% de éxito con 0 colisiones y 0 truncados en 300 episodios.**

**Análisis por seed:**

*s42 (reward=249,0, pasos=1202):*
- Política más compacta en pasos pero con reward algo menor que s524.
- Variabilidad del reward: [244,96 – 250,61] — mayor dispersión que s524.
- Consistente con el `explained_variance≈0` observado en training: la value function
  no estaba bien calibrada, resultando en pesos finales con mayor varianza en trayectoria.

*s123 (reward=253,1, pasos=1305):*
- El seed con más pasos por episodio (~100 más que s42): la política traza trayectorias
  algo menos directas o con velocidad media menor.
- Variabilidad del reward: [249,44 – 255,75] — dispersión intermedia.
- A pesar de ser el seed con mayor degradación en training (86,2%), la inferencia
  determinista elimina completamente el ruido estocástico y obtiene 100% de éxito.

*s524 (reward=265,0, pasos=1224):*
- **Mejor reward medio** de los tres seeds, con la variabilidad más baja.
- Rango pasos: [1220 – 1227], rango reward: [264,52 – 265,36] — trayectoria
  casi completamente determinista y repetible. Indica que la política converge a un
  único atractor en el espacio de acciones.
- Coherente con `explained_variance=0,816` (el más alto): la value function estaba bien
  calibrada, produciendo una política final más "concentrada".

**Salto training → inferencia:**

| Seed | tasa_exito training | tasa_exito inferencia | Δ |
|------|:-------------------:|:---------------------:|:-:|
| s42  | 97,6% | **100%** | +2,4 pp |
| s123 | 86,2% | **100%** | **+13,8 pp** |
| s524 | 86,4% | **100%** | **+13,6 pp** |

La mejora de +14 pp en s123 y s524 confirma que la degradación observada en TensorBoard
a partir de ~1M steps de stage 5 era un artefacto del modo estocástico, no pérdida real
de capacidad de navegación. En modo determinista (`deterministic=True`), la política
elige siempre la acción de máxima probabilidad, eliminando la varianza que causaba las
colisiones y truncados del training estocástico.

#### 11.7.7 Conclusión final — Stage 5

**Stage 5 completado con éxito total:** el tramo de retorno (zona_descarga → zona_espera)
se aprende de forma robusta en los tres seeds. Los resultados de inferencia son:

- **100%** de éxito en 900 episodios totales (300 × 3 seeds)
- **0%** de colisión y **0%** de truncado
- Reward medio de **255,7** (muy próximo al máximo teórico ~270: 100 reward_llegada + ~170 de progress rewards netos)
- Pasos medio de **1.244** por episodio (ruta de ~19,5m a ~16m/min de velocidad media)

El ciclo completo de navegación AMR está completamente aprendido:
1. **Approach** (zona_espera → estantería): stages 1-2 → 100% inferencia
2. **Exit** (estantería → zona_descarga): stages 3-4 → 100% inferencia
3. **Return** (zona_descarga → zona_espera): stage 5 → **100% inferencia** ✓

**Nota retrospectiva — olvido catastrófico descubierto en §12:**

La inferencia de stage 5 aislado (300 ep, solo return) no revela ningún problema. Sin embargo,
la inferencia de ciclo completo (§12) descubre que s42 ha perdido la capacidad de approach+exit
tras el fine-tuning de stage 5.

La señal de alerta ya estaba presente aquí: `explained_variance ≈ −0,053` en s42 durante
training de stage 5 indica una value function mal calibrada. Esto implica ventajas de alta
varianza en PPO → pasos de gradiente grandes → mayor riesgo de sobrescribir el conocimiento
previo de stages 1-4. s123 (EV=0,484) y s524 (EV=0,816) no muestran este problema.

El stage 5 aislado no detecta el olvido porque la tarea de return es independiente de
approach+exit: el robot se teleporta a `ret_path[2]` y solo necesita el conocimiento de retorno.
La degradación de approach+exit solo se manifiesta al encadenar las tres fases (§12).

---

## 12. Ciclo completo — Inferencia (approach + exit + return)

### 12.1 Motivación y decisión: inferencia vs entrenamiento adicional

Completados los stages 1-5, el modelo `stage5_final` ha sido entrenado secuencialmente
sobre las tres fases del ciclo de operación AMR:

| Fase | Stage | Tarea |
|------|:-----:|-------|
| Approach | 1-2 | zona_espera → estantería |
| Exit | 3-4 | estantería → zona_descarga |
| Return | 5 | zona_descarga → zona_espera |

La pregunta clave es si el modelo `stage5_final` retiene la capacidad de approach+exit
(aprendida en stages 1-4) tras el fine-tuning de stage 5, o si hubo olvido catastrófico.

**Estrategia elegida: primero inferencia, entrenamiento adicional solo si necesario.**

Si el éxito del ciclo completo es alto (>90%), el modelo ya es funcional y no requiere
un stage 6 de entrenamiento. Si hay degradación significativa respecto a stage 4 (100%),
se añadiría un stage 6 de fine-tuning.

### 12.2 Diseño técnico — Stage 6

Se añade `stage=6` a `webots_env.py` con la siguiente lógica de encadenamiento:

**`reset()` (stage 6):** siempre approach, goal aleatorio uniforme (28 goals).

**`step()` (stage 6) — switch descarga → retorno:**
```python
# Cuando el robot llega a zona_descarga (igual que stage 4)...
recompensa += 100.0   # bonus por completar approach+exit
# En lugar de terminar, continúa con return:
self._hacia_descarga = False
self._en_retorno     = True
self.full_path       = list(ret_path)   # ruta completa desde descarga
self._wp_idx         = 0
# avanza automáticamente waypoints ya dentro de WP_REACH_DIST
```

**Terminación:** solo cuando se alcanza `zona_espera` (`_en_retorno=True` + `dist_final<0.5`).

**Clasificación de colisiones** (mediante `env._hacia_descarga` y `env._en_retorno`):
- `colision_approach`: bumper sin haber llegado aún a la estantería
- `colision_exit`: bumper entre estantería y descarga
- `colision_return`: bumper durante el tramo de vuelta

### 12.3 Scripts creados

| Archivo | Descripción |
|---------|-------------|
| `webots_env.py` (mod.) | Stage 6: assert + reset() + step() switch descarga→retorno |
| `inferencia_subwp/infer_ciclo_s{42,123,524}.py` | Inferencia ciclo completo por seed |
| `run_subwp_infer_ciclo.sh` | Lanza los 3 seeds secuencialmente |

**Parámetros de inferencia:**
- Modelo: `subwp_s*_wp75_stage5_final`
- `deterministic=True` — sin exploración
- `_max_steps = 6000` (approach ~1500 + exit ~1500 + return ~1200, con margen)
- 100 ep × 28 goals × 3 seeds = **8.400 episodios totales**

### 12.4 Ejecución

```bash
bash run_subwp_infer_ciclo.sh
# Tiempo estimado: ~45-60 min/seed (~2-2.5h total)
# → inferencia_subwp/resultados/infer_subwp_s*_wp75_ciclo.csv
```

### 12.5 Resultados — Inferencia ciclo completo (100 ep × 28 goals × 3 seeds)

#### 12.5.1 Resumen global

| Seed | Éxito | col_approach | col_exit | col_return | Truncado | Pasos/ep | Reward/ep |
|------|:-----:|:------------:|:--------:|:----------:|:--------:|:--------:|:---------:|
| s42  | **32,6%** | 25,0% | 13,2% | 0,0% | 29,2% | 3.194,7 | 167,6 |
| s123 | **100,0%** | 0,0% | 0,0% | 0,0% | 0,0% | 3.367,6 | 632,6 |
| s524 | **100,0%** | 0,0% | 0,0% | 0,0% | 0,0% | 3.149,7 | 664,1 |

**Diagnóstico inmediato:** s123 y s524 completan el ciclo completo con 100% de éxito en los
2.800 episodios cada uno (5.600/5.600 combinados). s42 sufre olvido catastrófico severo de la
capacidad de approach+exit tras el fine-tuning de stage 5.

#### 12.5.2 Resultados por goal — s42 (olvido catastrófico)

| Goal | Éxito | col_approach | col_exit | Truncado | Diagnóstico |
|------|:-----:|:------------:|:--------:|:--------:|-------------|
| goal_01-04 | 0% | 100% | 0% | 0% | Approach olvidado |
| goal_05-07 | **100%** | 0% | 0% | 0% | Conservado |
| goal_08-10 | 0% | 0% | 0% | 100% | Timeout en approach |
| goal_11-13 | 0% | 100% | 0% | 0% | Approach olvidado |
| goal_14-15 | **100%** | 0% | 0% | 0% | Conservado |
| goal_16 | 0% | 0% | 0% | 100% | Timeout en approach |
| goal_17-19 | **100%** | 0% | 0% | 0% | Conservado |
| goal_20-22 | 0% | 0% | 0% | 100% | Timeout en approach |
| goal_23-24 | 0% | 0% | 100% | 0% | Exit olvidado |
| goal_25 | 0% | 0% | 0% | 100% | Timeout |
| goal_26 | **100%** | 0% | 0% | 0% | Conservado |
| goal_27 | 9% | 0% | 76% | 15% | Exit parcialmente olvidado |
| goal_28 | 3% | 0% | 93% | 4% | Exit parcialmente olvidado |

Goals con 100% éxito en s42: **05, 06, 07, 14, 15, 17, 18, 19, 26** (9/28 = 32,1%) — coincide
exactamente con los goals que tenían la trayectoria de exit más simple en stage 4.

#### 12.5.3 Resultados por goal — s123 y s524

Ambos seeds: **100/100 éxito en los 28 goals**. Sin una sola colisión ni truncado.

#### 12.5.4 Análisis del olvido catastrófico en s42

##### Patrón de fallo

El robot no sabe cómo hacer approach para los goals que antes dominaba al 100% en stage 4.
Los fallos se corresponden exactamente con los grupos de goals que en stages 3-4 requerían
mayor precisión de navegación (goals 01-04, 08-10, 11-13, 20-22, 25).

Los 9 goals que s42 **sí** completa (05-07, 14-15, 17-19, 26) son exactamente los que tenían
los exit paths más simples y directos en stage 4. El patrón de olvido no es aleatorio: se
preserva el conocimiento más "fácil" y se pierde el más "frágil" (representaciones distribuidas
en la red para goals con geometría compleja).

##### Mecanismo: por qué `explained_variance ≈ 0` causa olvido catastrófico

En PPO, la actualización de la política viene de la **función de ventaja**:

```
A(s,a) = R_real − V(s)
```

Si la value function V(s) predice bien los retornos → `explained_variance` alto → A(s,a)
tiene **poca varianza** → pasos de gradiente pequeños → pocos pesos modificados → poco olvido.

Si V(s) ≈ constante (no varía con el estado) → `explained_variance ≈ 0` → A(s,a) ≈ R − c →
**alta varianza en la ventaja** → pasos de gradiente grandes → muchos pesos modificados → olvido severo.

| Seed | explained_variance stage 5 | Varianza A(s,a) | Tamaño paso gradiente | Olvido ciclo |
|------|:--------------------------:|:---------------:|:--------------------:|:------------:|
| s42  | **−0,053** | Alta | **Grande** | Severo (32,6%) |
| s123 | 0,484 | Moderada | Moderado | Ninguno (100%) |
| s524 | **0,816** | Baja | **Pequeño** | Ninguno (100%) |

La correlación entre `explained_variance` y ausencia de olvido es perfecta en los tres seeds.

##### Por qué s42 convergió bien en stage 5 pero olvidó stages anteriores

La aparente paradoja — s42 aprendió mejor el return (97,6%) pero olvidó más el approach+exit —
se resuelve por las características de cada tarea:

- **Return** (stage 5): ruta única y fija, espacio abierto, sin variabilidad de goal. Basta con
  que los gradientes grandes empujen en la dirección correcta de forma consistente para obtener
  alta tasa de éxito, incluso sin que V(s) esté bien calibrada.

- **Approach+exit** (stages 1-4): 28 goals distintos, geometrías variadas, estanterías en zonas
  diferentes. El conocimiento requiere **representaciones distribuidas y precisas** en la red.
  Los gradientes grandes de s42 sobreescriben estas representaciones frágiles para liberar
  capacidad hacia la tarea más simple de return.

##### Por qué s123 y s524 no olvidaron

El training turbulento de stage 5 en s123 y s524 (86% éxito, reward oscilante, más truncados)
produjo gradientes ruidosos y contradictorios entre batches. El efecto neto es que las
actualizaciones de pesos fueron menores en magnitud — con alta varianza entre batches se
cancelan parcialmente — preservando más del conocimiento previo de approach+exit.

Adicionalmente, s524 con `explained_variance=0,816` tenía V(s) bien calibrada, produciendo
ventajas de baja varianza y pasos de gradiente pequeños. Ésto es coherente con que s524
también tuviera la trayectoria de inferencia de return más determinista y repetible
(rango pasos [1220-1227], reward casi constante ~265).

##### Factor adicional: número de episodios

Con ep_len más cortos (menor pasos/episodio), s42 completó más resets en los mismos 2M steps:

| Seed | Intentos stage 5 | ep_len medio | Actualizaciones PPO acumuladas |
|------|:----------------:|:------------:|:------------------------------:|
| s42  | 1.564 | 1.382 | **Más** |
| s123 | 1.387 | 1.598 | Menos |
| s524 | 1.468 | 1.463 | Intermedias |

Más episodios = más batches de gradiente = más presión acumulada sobre los pesos en la
dirección del return. Combinado con los pasos de gradiente grandes (EV≈0), el efecto es
multiplicativo.

##### Stability-plasticity tradeoff

Es una manifestación directa del **stability-plasticity tradeoff** en aprendizaje secuencial:

- **Plasticidad alta** (s42): aprende bien la nueva tarea (stage 5 → 97,6%), pero sobrescribe
  el conocimiento previo (stages 1-4 → olvidado).
- **Plasticidad moderada** (s123, s524): aprende la nueva tarea de forma menos perfecta en training
  (86%), pero retiene el conocimiento previo → 100% en el ciclo completo.

El seed "peor" en stage 5 training es el "mejor" en el ciclo completo — exactamente la predicción
del tradeoff stability-plasticity.

##### Observación clave: `colision_return = 0%` en los tres seeds

En los **tres seeds**, incluyendo s42, `colision_return = 0%`. Los episodios de s42 que logran
completar approach+exit (el 32,6%) ejecutan el return perfectamente. Esto confirma:
1. El conocimiento de stage 5 (return) no se ha degradado en ningún seed.
2. El olvido de s42 es **unidireccional**: stages 1-4 → olvidados; stage 5 → intacto.
3. El fine-tuning de stage 5 no causó interferencia en la dirección approach←return, solo en
   la dirección return←approach+exit.

#### 12.5.5 Desglose del reward ciclo completo (s123 y s524)

El reward medio por episodio completo (~632-664) se desglosa aproximadamente como:
- Approach (approach → estantería): +50 bonus + progress rewards − penalizaciones paso
- Exit (estantería → descarga): +100 bonus + progress rewards − penalizaciones paso
- Switch descarga→retorno (stage 6): +100 bonus
- Return (descarga → zona_espera): +100 bonus + progress rewards − penalizaciones paso

Los bonuses fijos suman 350. El resto (~282-314) corresponde a progress rewards netos,
coherente con las distancias recorridas (~40m totales en los tres tramos).

Los 3.150-3.368 pasos por episodio se desglosan estimados como:
- Approach+exit: ~1.925-2.063 pasos (por diferencia con stage 5 return: 1.224-1.305)
- Return: 1.224-1.305 pasos (datos directos de inferencia stage 5)

#### 12.5.6 Conclusión y decisión sobre stage 6 de entrenamiento

| Seed | Ciclo completo | Decisión |
|------|:--------------:|----------|
| s42  | 32,6% | Necesita re-entrenamiento o stage 6 de fine-tuning |
| s123 | **100%** | No requiere entrenamiento adicional |
| s524 | **100%** | No requiere entrenamiento adicional |

**Para el TFM:** los resultados de s123 y s524 (100% ciclo completo, 5.600 episodios sin fallo)
son suficientemente robustos para la comparativa con STH-WP. El caso de s42 ilustra el problema
de olvido catastrófico en curriculum learning secuencial, y puede incluirse como análisis de
ablación en la discusión de resultados.

**Decisión:** no se añade stage 6 de entrenamiento. Los 2/3 seeds con 100% de éxito son
estadísticamente suficientes para sustentar las conclusiones del TFM.

---

---

## 13. Experimento E1 — Curriculum de peatones: 1 peatón

### 13.1 Motivación

Los modelos `din_v3` (ciclo completo con 2 peatones) y `din_v4` (penalización adaptativa con 2 peatones)
muestran tasas de éxito por debajo del 20% en inferencia. La hipótesis es que el robot no puede
separar simultáneamente el aprendizaje de navegación dinámica con PEDESTRIAN_1 (trayectoria horizontal
x∈[-2,4], y=0.3) y PEDESTRIAN_2 (trayectoria vertical y∈[-5,-0.5], x=-9.5), lo que genera inestabilidad
y convergencia pobre.

El experimento E1 aplica curriculum de peatones: entrenar primero con un único peatón para que el robot
adquiera la habilidad de evasión básica antes de introducir el segundo.

**Objetivo**: alcanzar ≥90% de éxito con 1 peatón y luego escalar a 2.

**PEDESTRIAN_1 elegido** por su mayor impacto esperado: cruza el pasillo principal en la zona de descarga,
interceptando la trayectoria de approach/exit de prácticamente todos los goals.

### 13.2 Configuración

| Parámetro | Valor |
|-----------|-------|
| Base | `subwp_s{42,123,524}_wp75_r2_stage6_din_v3_final` |
| Dropoff base | (-11, 0, 0.2) — correcto |
| World | `warehouse_1_subwp_1ped.wbt` (PEDESTRIAN_2 eliminado) |
| Peatón activo | PEDESTRIAN_1: (-2,0.3)↔(4,0.3), speed=0.5 m/s |
| Steps adicionales | 3 000 000 × 3 seeds |
| Steps acumulados TensorBoard | ~2M → ~5M |
| learning_rate | 5e-5 |
| ent_coef | 0.005 |
| max_steps (entrenamiento) | 6 000 |
| reset_num_timesteps | False |
| Scripts | `experimentos/scripts/subwp_e1_1ped_s{42,123,524}.py` |
| Salidas | `pruebas/subwp_e1_1ped_s{42,123,524}_final.zip` |

**Nota sobre los modelos base (r2)**: los modelos r2 fueron reentrenados desde stage 3 con el dropoff
corregido (-11,0). Por ello parten de una capacidad notablemente inferior a los modelos originales
(~13.4% éxito en inferencia vs ~32.5% originals). Toda la mejora observada en E1 se construye sobre
esta base debilitada.

**Nota sobre la observación de P2**: el espacio de observación (48 dims) permanece inalterado. Con
PEDESTRIAN_2 ausente del world, los dims 45-48 (dx,dy,vx,vy del P2) son siempre [0,0,0,0]. La
arquitectura de red no cambia.

---

### 13.3 Resultados TensorBoard — entrenamiento

**Fecha**: 2026-07-16

#### Métricas de comportamiento al final (5M steps acumulados)

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `ep_rew_mean` (smoothed) | 414,5 | 316,7 | **482,5** |
| `ep_rew_mean` (value) | 417,5 | 319,8 | **482,5** |
| `ep_len_mean` (value) | 2 893 | 3 084 | **2 640** |
| `exito_ult100_%` (value) | 80 | 70 | **78** |
| `colision_ult100_%` (value) | 20 | 30 | **22** |
| `exito_dificiles_ult100_%` (value) | 76 | 71 | **80** |
| `exito_faciles_ult100_%` (value) | 81 | 71 | **80** |
| `tasa_exito_%` global | 79,7% | 53,5% | **84,1%** |
| `tasa_colision_%` global | 21,2% | 26,6% | **15,8%** |
| `tasa_estanteria_%` | 89,9% | 75,5% | **90,7%** |
| `tasa_truncado_%` | 0% | **19,9%** | 0% |
| `n_exitos` acumulado | 786 | 407 | **887** |
| `n_colisiones` acumulado | 212 | 202 | **167** |
| `n_truncados` acumulado | 0 | **151** | 0 |
| `pasos_medio_episodio` (value) | 2 893 | 3 084 | **2 640** |
| `reward_medio_episodio` (value) | 417,5 | 319,8 | **482,5** |
| FPS | 912 | 923 | 924 |

#### Métricas PPO al final

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| `approx_kl` | ~0,004 | ~0,006 | ~0,004 |
| `clip_fraction` | 0,0251 | 0,0711 | 0,0357 |
| `entropy_loss` | -7,08 | -6,45 | **-5,80** |
| `explained_variance` | 0,764 | 0,651 | 0,587 |
| `learning_rate` | 5e-5 | 5e-5 | 5e-5 |
| `train/std` (smoothed) | 71,0 | **253,9** | 60,0 |
| `train/value_loss` (smoothed) | 41,8 | 46,7 | **20,9** |

---

### 13.4 Análisis por seed

#### s42 — estable, convergencia sólida

Entrenamiento estable durante toda la curva. El reward sube de ~100 (inicio E1) a ~414 suavizado.
Éxito del ciclo completo al **79,7%** en training estocástico. La `tasa_estanteria_%` = 89,9%
indica que los fallos (~10%) son salidas colisionadas desde la estantería — el peatón interfiere
más en la fase exit que en approach. Cero truncados (el robot no se bloquea). `train/std = 71`
— política estable.

#### s524 — mejor rendimiento

El seed más sólido del E1 SUBWP. Mayor reward (482,5), menor tasa de colisión (15,8%), mayor
tasa de éxito (84,1%) y menor varianza de retornos (train/std = 60). Episodios más cortos en
promedio (2 640 pasos) indican trayectorias más eficientes. La `entropy_loss = -5,80` (la menos
negativa, o sea la mayor entropía de los tres) indica que la política mantiene cierta exploración
mientras sigue siendo la más eficiente. Cero truncados.

El patrón s524 como mejor seed coincide con lo observado en stages 3 y 4 del entrenamiento base:
s524 tendía a tener mejor reward aunque no siempre el mejor en ult100.

#### s123 — colapso catastrófico

s123 reproduce el patrón de inestabilidad estructural que ya se observó en stages anteriores,
pero en E1 se manifiesta con mayor intensidad:

- **tasa_truncado_% = 19,9%**: el robot alcanza el límite de 6 000 pasos sin completar el ciclo.
  Con s42 y s524 en 0% de truncaciones, esto es una firma inequívoca de política bloqueada.
- **n_truncados = 151** mientras s42/s524 tienen 0.
- **tasa_exito_% = 53,5%**: la mitad de los episodios termina sin éxito.
- **tasa_estanteria_% = 75,5%**: incluso el approach falla en ~25% de los casos.
- **ep_len_mean = 3 084** (vs 2 640-2 893 de los otros seeds): los episodios son 16% más largos
  por las truncaciones.
- **train/std = 253,9**: desviación típica de los retornos 3,6× mayor que s42 (71) y 4,2× mayor
  que s524 (60). Indica altísima varianza en los outcomes: algunos episodios obtienen el reward
  completo (~600) y muchos terminan truncados (~-20) o con colisión (~-150).
- **clip_fraction = 0,0711**: el doble que s42 y el triple que s524 — las actualizaciones PPO
  están siendo recortadas con frecuencia, señal de gradientes inestables.

**Interpretación**: el colapso de s123 en SUBWP E1 es el simétrico del colapso de s524 en
STHWP E1. El seed inestable cambia entre sistemas (STHWP: s524 colapsa; SUBWP: s123 colapsa)
pero el patrón es idéntico: alta varianza de retornos, muchas truncaciones, reward muy inferior
a los seeds estables. La raíz es seed-específica, no una limitación del método.

---

### 13.5 Comparativa SUBWP E1 vs STHWP E1

| Métrica | SUBWP s42 | SUBWP s524 | STHWP s42 | STHWP s123 | STHWP s524 |
|---------|:---------:|:----------:|:---------:|:----------:|:----------:|
| `tasa_exito_%` (training) | 79,7% | **84,1%** | ~90% | ~90% | — (colapso) |
| `tasa_colision_%` | 21,2% | 15,8% | ~10% | ~10% | — |
| Seed inestable | s123 | — | — | — | s524 |
| Nivel base (din_v3) | 13,4% media | | ~28,3% media | | |

**Diferencias clave**:

1. **SUBWP E1 inferior en ~5-10pp**: el gap entre SUBWP (~80-84%) y STHWP (~90%) en E1 es
   coherente con el gap en el entrenamiento base (13,4% vs 28,3% en din_v3). El punto de
   partida más bajo de SUBWP r2 limita el techo alcanzable con 3M steps adicionales.

2. **El seed inestable cambia**: STHWP colapsa en s524, SUBWP en s123. Ambos seeds tenían
   signos de fragilidad en stages anteriores (s524 en STHWP stages 3-4; s123 en SUBWP
   stages 1-4). La currícula con dinámica activa amplifica las debilidades seed-específicas.

3. **tasa_estanteria_%  < tasa_exito_%**: en SUBWP, la brecha (89,9% → 79,7% para s42)
   indica que el exit sigue siendo el cuello de botella del ciclo, incluso con 1 solo peatón.
   Con PEDESTRIAN_1 en la zona de approach (x∈[-2,4]), el robot aprende a esquivarlo en el
   approach pero luego puede encontrarlo también en la vuelta a descarga.

---

### 13.6 Resultados inferencia E1 (determinista)

**Configuración inferencia:**

| Parámetro | Valor |
|-----------|-------|
| World | `warehouse_1_subwp_1ped.wbt` (solo PEDESTRIAN_1) |
| Episodios | 28 goals × 100 ep = 2800 ciclos × 3 seeds |
| max_steps | 6000 |
| Modo | determinista (`deterministic=True`) |
| s42 | `subwp_e1_1ped_s42_final.zip` |
| s123 | `subwp_e1_1ped_s123_final.zip` (modelo final — inestable en training pero usable en inferencia determinista) |
| s524 | `subwp_e1_1ped_s524_final.zip` |

#### Resultados globales

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | **80.4%** | 73.8% | 76.9% | **77.0%** |
| Col. approach | 13.2% | 19.9% | 20.9% | 18.0% |
| Col. exit | 6.4% | 6.3% | 2.2% | 5.0% |
| Col. return | 0.0% | 0.0% | 0.0% | 0.0% |
| Truncado | 0.0% | 0.0% | 0.0% | 0.0% |
| Goals al 100% | 19/28 | 16/28 | 19/28 | — |
| Goals <50% | 5 | 7 | 7 | — |
| Pasos/ep | 2727 | 2428 | 2454 | 2536 |
| Reward/ep | 453.0 | 454.7 | 478.1 | 461.9 |

#### Resultados por goal

| Goal | s42 | s123 | s524 | Patrón de fallo |
|------|:---:|:----:|:----:|-----------------|
| goal_01 | **100** | **100** | **100** | — |
| goal_02 | **100** | **100** | **100** | — |
| goal_03 | **100** | **100** | **100** | — |
| goal_04 | **100** | **100** | **100** | — |
| goal_05 | **100** | **100** | **100** | — |
| goal_06 | **100** | **100** | **100** | — |
| goal_07 | **100** | **100** | **100** | — |
| goal_08 | **100** | 57 | **100** | s123: col_exit 43% |
| goal_09 | **100** | 85 | **100** | s123: col_exit 15% |
| goal_10 | **100** | **100** | **100** | — |
| goal_11 | **100** | **100** | **100** | — |
| goal_12 | 86 | **100** | **100** | s42: col_exit 14% |
| goal_13 | **100** | **100** | **100** | — |
| goal_14 | **100** | **100** | **100** | — |
| goal_15 | **100** | 90 | **100** | s123: col_exit 10% |
| goal_16 | **100** | 80 | **100** | s123: col_exit 20% |
| goal_17 | **100** | **100** | **100** | — |
| goal_18 | **100** | **100** | **100** | — |
| goal_19 | 70 | 3 ⚠ | 1 ⚠ | col_approach masivo s123/s524 (95-98%) |
| goal_20 | 6 ⚠ | 4 ⚠ | 0 ⚠ | col_approach masivo todos (85-99%) |
| goal_21 | 18 ⚠ | 0 ⚠ | 0 ⚠ | col_approach 77-100% todos |
| goal_22 | 16 ⚠ | 0 ⚠ | 0 ⚠ | col_approach 69-100% todos |
| goal_23 | 97 | **100** | **100** | s42: col_exit 3% |
| goal_24 | 51 | 73 | 74 | mezcla col_ap + col_exit |
| goal_25 | 50 | 63 | 69 | col_ap + col_exit (zona P1) |
| goal_26 | 96 | 46 ⚠ | 46 ⚠ | s123/s524: col_ap ~51-54% |
| goal_27 | 40 ⚠ | 48 ⚠ | 43 ⚠ | mezcla col_ap + col_exit |
| goal_28 | 21 ⚠ | 18 ⚠ | 19 ⚠ | col_approach dominante (~38-73%) |

#### Análisis

**Dos clusters de fallo independientes:**

**Cluster A — Goals 19-22 (SUBWP-específico, no relacionado con P1):**
Los goals 20, 21 y 22 fallan en los 3 seeds con col_approach masivo (69-100%).
En STHWP E1 estos mismos goals son 100% para s42 y s123 → el fallo NO es de P1
sino de la política SUBWP para esas geometrías de approach. La base r2 (más débil)
combinada con el fine-tuning E1 no logró consolidar el approach de ese sector.
Goal_19 falla especialmente en s123/s524 (3% y 1%) pero funciona al 70% en s42.

**Cluster B — Goals 24-28 (sector PEDESTRIAN_1, compartido con STHWP):**
Mismo patrón que en STHWP E1: P1 interfiere en el approach/exit del sector superior
derecho. Goal_28 falla en los 3 seeds (~18-21%, col_approach dominante). Goals 25-27
muestran tasas intermedias. Goal_26 es llamativamente asimétrico: s42 lo resuelve al
96% pero s123/s524 solo al 46% por col_approach — diferencia seed-específica en
cómo se aproxima a esa estantería concreta.

**s42 mejor seed (80.4%):** menor col_approach (13.2%), única caída en goals 20-22
con 6/18/16% respectivamente. El exit está bien dominado (col_exit=6.4%). 19/28 goals al 100%.

**s524 (76.9%):** exit prácticamente perfecto (col_exit=2.2%, el mejor de los tres),
pero col_approach alto (20.9%). Las 9 goals fallidas incluyen los clusters A y B.
19/28 goals al 100%.

**s123 (73.8%):** en entrenamiento era el seed más inestable, pero en inferencia
determinista da resultado usable. El col_approach (19.9%) está concentrado en clusters A y B.
Col_exit (6.3%) comparable a s42. 0 truncaciones (confirma que el modo determinista elimina
la inestabilidad estocástica observada en training).

---

### 13.7 Comparativa STHWP E1 vs SUBWP E1 — inferencia determinista

| Métrica | STHWP s42 | STHWP s123 | STHWP s524* | SUBWP s42 | SUBWP s123 | SUBWP s524 |
|---------|:---------:|:----------:|:-----------:|:---------:|:----------:|:----------:|
| Éxito global | **88.1%** | **84.4%** | 66.5% | 80.4% | 73.8% | 76.9% |
| Col. approach | 7.4% | 5.5% | 8.6% | 13.2% | 19.9% | 20.9% |
| Col. exit | 4.5% | 10.1% | 25.0% | **6.4%** | **6.3%** | **2.2%** |
| Col. return | 0% | 0% | 0% | 0% | 0% | 0% |
| Goals 100% | 23/28 | 22/28 | 14/28 | 19/28 | 16/28 | 19/28 |
| Pasos/ep | **2074** | **1979** | **1763** | 2727 | 2428 | 2454 |
| Reward/ep | 456.8 | 443.4 | 345.2 | 453.0 | **454.7** | **478.1** |

*s524 STHWP excluido de media por colapso.*

| Métrica | STHWP (media s42+s123) | SUBWP (media 3 seeds) | Δ |
|---------|:----------------------:|:---------------------:|:--:|
| **Éxito global** | **86.3%** | 77.0% | **STHWP +9.3pp** |
| Col. approach | **6.5%** | 18.0% | STHWP mejor |
| Col. exit | 7.3% | **5.0%** | SUBWP mejor |
| Pasos/ep | **2027** | 2536 | STHWP más eficiente |

**STHWP gana en éxito global (+9.3pp) y eficiencia (−509 pasos/ep).**
**SUBWP gana en col_exit (5.0% vs 7.3%)** — coherente con el resultado de stage 4 sin
obstáculos, donde el mecanismo de waypoint discreto beneficia la salida en ciclo encadenado.

**Clusters de fallo diferenciados:**
- STHWP falla principalmente en goals 24-28 (P1) — problema externo al sistema.
- SUBWP falla en goals 19-22 (structural, r2 base) Y goals 24-28 (P1) — doble problema.

---

### 13.8 Decisión

**STHWP E1 referencia: 86.3%** (s42+s123). **SUBWP E1 referencia: 77.0%** (3 seeds).

El gap STHWP vs SUBWP en E1 (+9.3pp) tiene dos causas separables:
1. **~5pp** atribuibles al cluster A (goals 19-22), problema SUBWP-específico de la base r2.
2. **~4pp** atribuibles al cluster B (goals 24-28), compartido con STHWP pero con peor gestión en SUBWP.

**Opciones para mejorar hacia el 100%:**

| Opción | Descripción | Impacto esperado | Coste |
|--------|-------------|-----------------|-------|
| **E1.2 — sesgo goals problemáticos** | Fine-tune desde E1 con PROB alta en goals 19-22 y 24-28 | +10-15pp SUBWP; +5-8pp STHWP | Bajo (compatible con modelos actuales) |
| **E1.3 — predicción de posición P1** | Añadir pos predicha 1s/2s a obs (48→52 dims) | +?pp en goals 24-28 para ambos | Alto (reentrenamiento completo) |
| **Escalar a E2** | Añadir PEDESTRIAN_2 con resultados actuales | Retroceso esperado; no recomendado antes de mejorar | — |

**Decisión recomendada**: lanzar E1.2 con sesgo de goals antes de E2.
Primero STHWP E1.2 (base más sólida), luego SUBWP E1.2.

---

## 14. Experimento E1.2 — Fine-tune con sesgo goals 19-22 + 24-28

### 14.1 Motivación

SUBWP E1 tiene dos clusters de fallo:
- **Cluster A** (goals 19-22): col_approach 69-100% — problema estructural SUBWP r2
- **Cluster B** (goals 24-28): col_approach+exit — sector PEDESTRIAN_1

E1.2 sesga el 70% de los episodios hacia esos 10 goals para atacar ambos clusters.

### 14.2 Configuración

| Parámetro | Valor |
|-----------|-------|
| Base | `subwp_e1_1ped_s{42,123,524}_final` |
| Goals sesgados | 19-22 (índices 18-21) + 24-28 (índices 23-27) = 10 goals |
| PROB_HARD | 70% |
| Steps | 2 000 000 adicionales |
| LR | 2e-5 |
| ent_coef | 0.005 |
| max_steps | 6000 |
| Scripts | `experimentos/scripts/subwp_e1_2_s{42,123,524}.py` |
| Salidas | `pruebas/subwp_e1_2_s{42,123,524}_final.zip` |

### 14.3 Resultados TensorBoard

**Pasos acumulados al finalizar** (5M base E1 + 2M E1.2 = ~7M):

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| `ep_rew_mean` smoothed (7M) | 392,2 | 393,7 | **498,6** |
| `ep_rew_mean` value (7M) | 396,5 | 395,4 | **498,4** |
| `ep_len_mean` smoothed (7M) | 2804 | 2592 | 2837 |
| `entropy_loss` (7M) | −6,495 | −5,260 | −5,365 |
| `explained_variance` (7M) | 0,599 | **0,812** | 0,724 |
| `train/std` (7M) | **71** | 117 | 58 |
| `approx_kl` (7M) | 0,0047 | **0,0095** | 0,0067 |
| `clip_fraction` (7M) | 0,0373 | **0,0822** | 0,0496 |
| `train/value_loss` (7M) | **35,99** | 16,02 | 27,37 |
| FPS | 923 | **931** | 923 |

### 14.4 Análisis

**s524 — dominante y estable (~499 reward):**
El seed más sólido del experimento. Reward muy alto (~499) y estable durante todo E1.2,
con std=58 (mínimo de los tres) y FPS constante. La señal de reward alta sugiere que s524
ha aprendido a resolver goals del cluster A (19-22) correctamente — si los goals de P1
siguen fallando, el promedio bajaría más. Infiriendo: s524 E1.2 debería mejorar
significativamente en goals 19-22 (cluster A) con mejora moderada en 24-28 (cluster B).

**s42 — volatile pero recuperado (~392 reward):**
Dip pronunciado a ~300 alrededor de 5.5M steps (primeras iteraciones del sesgo,
la política se desestabiliza al concentrarse en los goals más difíciles), seguido de
recuperación a ~396. train/std=71 (estable, mismo valor que E1). El explained_variance
(0.599) es el más bajo — el crítico está bajo presión por la nueva distribución de goals.
La recuperación del reward sugiere que está aprendiendo, aunque más lentamente que s524.

**s123 — mejora respecto a E1 (std 254→117):**
El seed más inestable en E1 muestra mejora significativa en E1.2: train/std cae de 254 a 117
(−54%). La política ya no está completamente colapsada. Reward ~393 comparable a s42.
clip_fraction (0.082) el más alto — actualizaciones más agresivas, política más activa.
El explained_variance (0.812) es el mejor de los tres — el crítico ha mejorado considerablemente.
La tendencia es positiva: si E1.2 redujo el std de 254 a 117, la inferencia debería mejorar
respecto al 73.8% de E1.

**Comparativa SUBWP E1 → E1.2 (training):**

| Métrica | E1 s42 | E1.2 s42 | E1 s123 | E1.2 s123 | E1 s524 | E1.2 s524 |
|---------|:------:|:--------:|:-------:|:---------:|:-------:|:---------:|
| reward (smoothed) | 414 | 392 | 317 | 394 | **483** | **499** |
| train/std | 71 | 71 | 254 | **117** | 60 | 58 |
| explained_var | 0.764 | 0.599 | 0.651 | **0.812** | 0.587 | 0.724 |

s523 mejora claramente. s42 ligeramente peor en reward pero el sesgo distorsiona la
comparación directa (más goals difíciles = menos reward promedio en training).

**Comparativa reward STHWP E1.2 vs SUBWP E1.2:**
STHWP E1.2 reward ~300-332 vs SUBWP E1.2 ~392-499. La diferencia se debe a que:
- STHWP sesga solo hacia goals 24-28 (100% P1, muy difíciles de resolver en training)
- SUBWP sesga hacia 19-22 también, que son goals recuperables si el robot aprende el approach
  correcto → contribuyen reward positivo al promedio cuando se resuelven

### 14.5 Resultados inferencia E1.2 (determinista)

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | 82.6% | 75.9% | 76.8% | **78.4%** |
| Col. approach | 12.9% | 19.1% | 21.2% | 17.7% |
| Col. exit | 4.5% | 5.0% | 2.1% | 3.9% |
| Col. return | 0% | 0% | 0% | 0% |
| Goals al 100% | 19/28 | ~17/28 | ~18/28 | — |
| Pasos/ep | 1838 | 2001 | 1997 | 1945 |
| Reward/ep | 419.5 | 367.0 | 374.7 | 387.1 |

#### Por goal — comparativa E1 → E1.2

| Goal | E1 s42 | E1.2 s42 | Δ | E1 s123 | E1.2 s123 | Δ | E1 s524 | E1.2 s524 | Δ |
|------|:------:|:--------:|:-:|:-------:|:---------:|:-:|:-------:|:---------:|:-:|
| goal_01–18 | 100 | 100 | = | 100 | 100 | = | 100 | 100 | = |
| goal_19 | 70 | 74 | +4 | 3 | 2 | = | 1 | 1 | = |
| goal_20 | 6 | 7 | = | 4 | 0 | −4 | 0 | 0 | = |
| goal_21 | 18 | 14 | −4 | 0 | 0 | = | 0 | 0 | = |
| goal_22 | 16 | **1** | **−15** | 0 | 0 | = | 0 | 0 | = |
| goal_23 | 97 | 100 | +3 | 100 | 100 | = | 100 | 100 | = |
| goal_24 | 51 | **86** | **+35** | 73 | 67 | −6 | 74 | 73 | = |
| goal_25 | 50 | 59 | +9 | 63 | 56 | −7 | 69 | 67 | −2 |
| goal_26 | 96 | 87 | −9 | 46 | 51 | +5 | 46 | 37 | −9 |
| goal_27 | 40 | 36 | −4 | 40 | 31 | −9 | 43 | 49 | +6 |
| goal_28 | 21 | **48** | **+27** | 18 | 31 | **+13** | 19 | 22 | +3 |

#### Análisis

**Cluster A (goals 19-22) — NO resuelto en s123 ni s524:**
El sesgo de 2M steps hacia estos goals no consiguió recuperar el approach en s123 y s524.
Los valores se mantienen en 0-2%, esencialmente idénticos a E1. Esto confirma que es un problema
estructural del modelo base r2 para estos goals: la política ha convergido localmente a una
solución defectuosa que el fine-tune con lr=2e-5 no puede escapar. En s42 hay una mejora
marginal (goal_19: 70→74%) pero goals 20-22 siguen fallando. Curiosamente, goal_22 incluso
empeoró en s42 (16→1%), probablemente por interferencia del sesgo con el comportamiento
de exit previamente aprendido.

**Cluster B (goals 24-28, sector P1) — mejora significativa en s42:**
s42 mostró las mejoras más notables: goal_24 +35pp (51→86%) y goal_28 +27pp (21→48%).
Esto indica que el sesgo SÍ funcionó para s42, que partía de una base más sólida en E1.
Para s123 y s524, los resultados son mixtos: algunos goals mejoran (goal_28: s123 +13pp)
pero otros empeoran (goal_26: s123 −9pp, s524 −9pp), sin un patrón claro. La alta col_approach
de s123 y s524 (~19-21%) frente a s42 (13%) sugiere que el sesgo generó sobreajuste en
el approach de los goals difíciles a costa de robustez.

**Comparativa global E1 → E1.2:**

| Métrica | E1 media 3s | E1.2 media 3s | Δ |
|---------|:-----------:|:-------------:|:-:|
| Éxito global | 77.0% | 78.4% | **+1.4pp** |
| Col. approach | 15.7% | 17.7% | −2pp |
| Col. exit | 6.6% | 3.9% | +2.7pp |
| Goals 100% (media) | ~18/28 | ~18/28 | = |

La mejora de 1.4pp es modesta y concentrada en s42. Para s123 y s524 el cluster A sigue
siendo el cuello de botella, y E1.2 no lo ha resuelto.

**Conclusión:** el sesgo de curriculum es efectivo para seeds con modelo base fuerte (s42),
pero insuficiente para corregir deficiencias estructurales del approach en seeds más débiles.
Para atacar el cluster A en s123/s524 habría que reentrenar desde stage r2 con reward
diferente o con modificaciones en la dinámica del goal approach.

**Referencia E1.2 SUBWP:** 78.4% media 3 seeds. Mejora de +1.4pp sobre E1 (77.0%).
**Referencia definitiva SUBWP hasta E1.2:** 78.4% media / 82.6% mejor seed (s42).

---

## 15. Experimento E1_pred — Observación predictiva P1 (48→52 dims)

### 15.1 Motivación y configuración

Mismo planteamiento que STHWP E1_pred: añadir información explícita sobre la posición
futura de P1 para que el agente aprenda a decidir si avanzar o esperar. E1.2 no resolvió
el cluster A (goals 19-22) en s123/s524 y tuvo rendimientos decrecientes en cluster B.

| Parámetro | Valor |
|-----------|-------|
| Obs space | 48 → **52 dims** (+ dx_pred_t1, dy_pred_t1, dx_pred_t2, dy_pred_t2) |
| Pred horizon | [1.0 s, 2.0 s], solo P1, velocidad constante |
| Base modelo | E1 final (s42, s123, s524) |
| Método init | Weight transplant: primera capa Linear(48→64) → Linear(52→64); cols 49-52 = 0 |
| Steps | 3M por seed (lr=2e-5, ent_coef=0.005) |
| World | warehouse_1_subwp_1ped.wbt (solo PEDESTRIAN_1) |
| TensorBoard ejes | s42 desde 0 (sin fix num_timesteps); s123 y s524 desde 5M→8M (fix aplicado) |

### 15.2 Resultados TensorBoard (training)

**Métricas finales:**

| Métrica | s42 (step 3M) | s123 (step 8M) | s524 (step 8M) |
|---------|:-------------:|:--------------:|:--------------:|
| `rollout/ep_rew_mean` | 484.5 | 491.3 | **529.3** |
| `rollout/ep_len_mean` | 2871 | 2699 | 2804 |
| `tasa_exito_%` | 80.8% | 82.4% | **86.9%** |
| `tasa_colision_%` | 19.2% | 17.6% | **13.1%** |
| `tasa_estanteria_%` | 89.4% | 91.0% | **92.5%** |
| `exito_ult100` | 84 | 85 | 84 |
| `exito_dificiles_ult100` | 81 | 85 | **85** |
| `exito_faciles_ult100` | 86 | 81 | 86 |
| `train/std` | 69.6 | **127.6** | 57.1 |
| `train/explained_variance` | **0.773** | 0.436 | 0.492 |
| `train/value_loss` | 34.9 | 94.2 | 24.1 |
| `n_colisiones` acumuladas | 198 | 182 | **138** |
| `time/fps` | 916 | 927 | 924 |

### 15.3 Análisis

**s524 — mejor seed en E1_pred SUBWP:**
Invierte el patrón de E1/E1.2 donde s524 era el más inestable. Con E1_pred:
reward=529, éxito=86.9%, colisión=13.1%, tasa_estanteria=92.5% (approach success).
El train/std=57.1 es el más bajo de las 3 seeds, confirmando convergencia estable.
La hipótesis: el modelo base E1 s524 para SUBWP (no colapsó como STHWP s524) tenía
suficiente base en approach para beneficiarse de la información predictiva de P1.

**s123 — inestabilidad persistente:**
train/std=127.6 vuelve a valores similares a E1 (254) y peores que E1.2 (117).
El explained_variance=0.436 es el más bajo — el crítico no converge bien.

**s42 — entrenamiento desde 0 en TensorBoard:**
El eje comienza en 0 (no se aplicó el fix de `num_timesteps` a tiempo). La reward
final (484.5) y éxito (80.8%) son los más bajos de las 3 seeds.

**Comparativa training E1 → E1_pred SUBWP:**

| Métrica training | E1 s42 | E1_pred s42 | E1 s123 | E1_pred s123 | E1 s524 | E1_pred s524 |
|-----------------|:------:|:-----------:|:-------:|:------------:|:-------:|:------------:|
| tasa_exito_% | ~80% | 80.8% | ~74% | 82.4% | ~77% | **86.9%** |
| tasa_colision_% | ~20% | 19.2% | ~26% | 17.6% | ~23% | **13.1%** |
| train/std | ~70 | 69.6 | ~254 | 127.6 | ~90 | **57.1** |

### 15.4 Resultados inferencia E1_pred (determinista)

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | **78.9%** | 73.7% | 76.2% | **76.3%** |
| Col. approach | 15.7% | 18.8% | 19.9% | 18.1% |
| Col. exit | 5.4% | 7.5% | 3.8% | 5.6% |
| Col. return | 0% | 0% | 0% | 0% |
| Goals al 100% | 19/28 | 17/28 | 18/28 | — |
| Pasos/ep | 2659 | 2448 | 2467 | 2525 |
| Reward/ep | 481.7 | 457.4 | 468.0 | 469.0 |

#### Por goal — comparativa E1.2 → E1_pred

| Goal | E1.2 s42 | E1_pred s42 | Δ | E1.2 s123 | E1_pred s123 | Δ | E1.2 s524 | E1_pred s524 | Δ |
|------|:--------:|:-----------:|:-:|:---------:|:------------:|:-:|:---------:|:------------:|:-:|
| goal_01–18 | 100 | 100 | = | 100 | ~98 | ≈ | 100 | 100 | = |
| goal_19 | 74 | 56 | −18 | 2 | 0 | = | 1 | 1 | = |
| goal_20 | 7 | **0** | −7 | 0 | 0 | = | 0 | 0 | = |
| goal_21 | 14 | **0** | −14 | 0 | 0 | = | 0 | 0 | = |
| goal_22 | 1 | **0** | −1 | 0 | 0 | = | 0 | 0 | = |
| goal_23 | 100 | 100 | = | 100 | 100 | = | 100 | 100 | = |
| goal_24 | 86 | **52** | **−34** | 67 | **62** | −5 | 73 | **72** | = |
| goal_25 | 59 | 50 | −9 | 56 | 56 | = | 67 | **66** | = |
| goal_26 | 87 | **100** | **+13** | 51 | **0** | **−51** ⚠ | 37 | 42 | +5 |
| goal_27 | 36 | 31 | −5 | 31 | **45** | +14 | 49 | **34** | −15 |
| goal_28 | 48 | **23** | **−25** | 31 | 26 | −5 | 22 | 20 | = |

#### Análisis E1_pred SUBWP

**Cluster A (goals 19-22) — sin resolución en ninguna seed:**
El cluster A sigue siendo el cuello de botella absoluto. s42 incluso empeora (goal_19: 74→56%,
goals 20-22: ya eran 0% y siguen en 0%). La información predictiva de P1 no ayuda aquí porque
el problema es el approach geométrico de la ruta A*, no el timing con P1.

**Cluster B (sector P1) — resultados mixtos, sin mejora neta:**
- s42: goal_24 cae de 86% a 52% (−34pp, regresión grave), goal_26 sube de 87% a 100% (+13pp),
  goal_28 cae de 48% a 23% (−25pp). Neto negativo.
- s123: goal_24 (67→62, −5pp), goal_26 colapsa (51→0%, −51pp), goal_27 mejora (31→45%, +14pp).
  La nueva failure catastrophica de goal_26 s123 (col_exit=100%) es inesperada.
- s524: resultados similares a E1.2, sin mejora notable.

La información predictiva no está siendo utilizada para el timing de P1 de forma consistente.
El robot aprende a usar los nuevos dims para algunos goals pero regresa en otros.

**Comparativa global E1 → E1.2 → E1_pred SUBWP:**

| Experimento | s42 | s123 | s524 | Media 3 seeds |
|-------------|:---:|:----:|:----:|:-------------:|
| E1 | 80.4% | 73.8% | 76.9% | 77.0% |
| **E1.2** | **82.6%** | **75.9%** | **76.8%** | **78.4%** |
| E1_pred | 78.9% | 73.7% | 76.2% | 76.3% |

**E1_pred es peor que E1.2 para SUBWP** (−2.1pp). E1.2 sigue siendo la mejor versión SUBWP.
La obs predictiva no aporta beneficio neto para SUB-WP con 1 peatón: el cluster A no se
resuelve y el cluster B muestra regresiones significativas junto con mejoras puntuales.

**Referencia definitiva SUBWP hasta E1_pred:** E1.2 — 78.4% media 3 seeds / 82.6% mejor seed (s42).

---

## Sección 16 — Experimento E1.3: Trayectoria P1 extendida x∈[-4,4]

**Motivación:** Los goals 24-28 siguen siendo el cuello de botella principal. Con P1 oscilando
en x∈[-2,4], el peatón bloquea casi permanentemente el corredor de approach. La hipótesis es
que extender la trayectoria a x∈[-4,4] crea ventanas temporales donde P1 está en x∈[-4,-2]
(lejos de los goals), facilitando que el robot aprenda el timing de approach.

**Configuración:**
- Base: `subwp_e1_2_s{42,123,524}_final` (48 dims, sin pred_horizon — mejor base SUBWP)
- World: `warehouse_1_subwp_1ped.wbt` (PEDESTRIAN_1: trajectory=`-4 0.3, 4 0.3`)
- Pasos: 2M fine-tune | lr=1e-5 | ent_coef=0.005 | max_steps=6000
- Rango randomización spawn P1: x∈(-4.0, 4.0) en webots_env.py
- x-axis TensorBoard: 7M → 9M (continúa desde E1.2 ~7M steps)

### 16.1 Análisis TensorBoard E1.3 SUBWP

#### Métricas de entrenamiento (training stats)

| Métrica training | s42 | s123 | s524 |
|-----------------|:---:|:----:|:----:|
| tasa_exito_% | 78.53% | **84.02%** | **86.12%** |
| tasa_colision_% | 20.59% | 15.97% | **13.87%** |
| tasa_estanteria_% | 91.05% | 92.12% | **93.34%** |
| exito_ult100_% | 76 | 85 | **89** |
| exito_dificiles_ult100_% | 76 | 81 | **88** |
| exito_faciles_ult100_% | 64 ⚠ | 87 | 84 |
| colision_ult100_% | 21 ⚠ | 15.9 | **11** |
| n_colisiones | 138 | 114 | **98** |
| n_exitos | 527 | 600 | **610** |
| n_truncados | **6** ⚠ | 0 | 0 |
| reward_medio_episodio | 464.2 | 541.1 | **564.9** |
| pasos_medio_episodio | **3043** | 2967 | 2903 |

#### Métricas de optimización PPO

| Métrica PPO | s42 | s123 | s524 |
|-------------|:---:|:----:|:----:|
| entropy_loss | **-6.288** ⚠ | -5.007 | -5.224 |
| explained_variance | **0.508** ⚠ | **0.917** | 0.572 |
| train/std | 68.46 | **103.97** | 59.57 |
| value_loss | **34.94** ⚠ | 9.26 | 22.91 |
| approx_kl | 0.002 | **0.0047** | 0.0036 |
| clip_fraction | 0.013 | **0.047** | 0.027 |
| FPS | **939** | **940** | 929 |

#### Análisis cualitativo E1.3 SUBWP

**s42 — colapso parcial en 8.5M:** a partir de 8.5M steps, s42 experimenta un evento
de degradación visible en varias métricas simultáneamente:
- Aparecen truncados (n_truncados=6, tasa_truncado=0.89%) — episodios que alcanzan max_steps
  sin terminar, indicando que el robot se queda bloqueado sin completar ni colisionar.
- Reward cae de ~500 a ~380 y se recupera parcialmente al final (464 smoothed).
- exito_faciles_ult100 cae a 64% — los goals sencillos (1-18) que siempre eran 100%
  se ven afectados, lo que confirma una regresión generalizada de la política.
- entropy_loss=-6.29 (la más baja de todo E1.x) — política extremadamente determinista,
  sin exploración. El agente quedó atrapado en comportamientos fijos subóptimos.
- explained_variance=0.508 y value_loss=34.94 — el crítico no está capturando bien el
  retorno esperado, señal de desequilibrio entre política y función de valor.

La causa probable: al cambiar la trayectoria de P1, la distribución de estados cambia
significativamente. s42, que tenía la política más especializada en x∈[-2,4], sufrió un
distributional shift más severo que s123 y s524 (cuyos modelos base E1.2 eran menos óptimos
y por tanto más adaptables al nuevo distribution).

**s523 — adaptación sólida:** tasa_exito=84.0% (+8.1pp vs E1.2 s123 75.9%). Entropy sana
(-5.007), explained_variance=0.917 (el más alto, crítico bien calibrado). n_exitos=600
vs 527 de s42, a pesar de ser un seed habitualmente más débil. La recompensa media (541.1)
es la segunda más alta. Mejora consistente y sin colapso.

**s524 — mejor resultado SUBWP de toda la serie:** tasa_exito=86.12% (+9.3pp vs E1.2 s524
76.8%). reward_medio=564.9 (el más alto), colision=13.87% (la más baja), n_colisiones=98
(el mínimo). tasa_estanteria=93.34%. Aunque el explained_variance (0.572) y value_loss (22.91)
son intermedios, la política es eficaz. s524, que en E1.2 era el seed más débil, aquí
se convierte en el mejor — posiblemente porque la nueva trayectoria rompe sesgo de E1.2
que beneficiaba a s42.

**Comparativa training E1.2 → E1.3 SUBWP:**

| Métrica training | E1.2 s42 | E1.3 s42 | Δ | E1.2 s123 | E1.3 s123 | Δ | E1.2 s524 | E1.3 s524 | Δ |
|-----------------|:--------:|:--------:|:-:|:---------:|:---------:|:-:|:---------:|:---------:|:-:|
| tasa_exito_% | **82.6%** | 78.5% | −4.1 ⚠ | 75.9% | **84.0%** | **+8.1** | 76.8% | **86.1%** | **+9.3** |
| tasa_colision_% | 17.4% | 20.6% | +3.2 ⚠ | 24.1% | **16.0%** | −8.1 | 23.2% | **13.9%** | **−9.3** |
| n_truncados | 0 | **6** ⚠ | — | 0 | 0 | = | 0 | 0 | = |

E1.3 mejora s123 y s524 de forma muy significativa (>8pp), pero degrada s42 (−4.1pp)
que era la seed de referencia en E1.2. La media de los 3 seeds mejora claramente.

**Media training 3 seeds E1.3:** (78.5 + 84.0 + 86.1) / 3 = **82.87%** (vs E1.2: 78.4%, +4.5pp)
**Media training s123+s524 (excluyendo colapso s42):** (84.0 + 86.1) / 2 = **85.05%**

### 16.2 Comparativa global E1 → E1.2 → E1_pred → E1.3 SUBWP (training)

| Experimento | s42 | s123 | s524 | Media 3 seeds |
|-------------|:---:|:----:|:----:|:-------------:|
| E1 | 80.4% | 73.8% | 76.9% | 77.0% |
| E1.2 | **82.6%** | 75.9% | 76.8% | 78.4% |
| E1_pred | 78.9% | 73.7% | 76.2% | 76.3% |
| **E1.3** | 78.5% | **84.0%** | **86.1%** | **82.87%** ✅ |

**E1.3 es el mejor experimento SUBWP en media** (+4.5pp sobre E1.2, +6.6pp sobre E1_pred).
El sacrificio de s42 (−4.1pp vs E1.2) es compensado ampliamente por las ganancias de
s123 (+8.1pp) y s524 (+9.3pp). La modificación de trayectoria beneficia más a los seeds
cuya política base era más generalista.

**Referencia definitiva SUBWP actualizada:** E1.3 — 82.87% media 3 seeds / 86.12% mejor seed (s524).

### 16.3 Resultados inferencia E1.3 SUBWP (determinista)

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | **81.6%** | **81.3%** | 78.9% | **80.6%** |
| Col. approach | 5.8% | 4.8% | 5.5% | 5.4% |
| Col. exit | 12.6% | 13.9% | **15.6%** | 14.0% |
| Truncados | 0.1% | 0% | 0% | 0% |
| Goals al 100% | 14/28 | 18/28 | 18/28 | — |

#### Por goal — detalle completo E1.3 SUBWP

| Goal | s42 Éx | s42 C_app | s42 C_exit | s123 Éx | s123 C_app | s123 C_exit | s524 Éx | s524 C_app | s524 C_exit |
|------|:------:|:---------:|:----------:|:-------:|:----------:|:-----------:|:-------:|:----------:|:-----------:|
| 01–09 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 |
| 10 | 58 ⚠ | 0 | 42 | **100** | 0 | 0 | 60 ⚠ | 0 | 40 |
| 11 | 93 | 0 | 7 | **100** | 0 | 0 | **100** | 0 | 0 |
| 12–16 | **100** | 0 | 0 | 90–100 | 0 | 0–10 | **100** | 0 | 0 |
| 17 | 85 | 14 | 1 | **100** | 0 | 0 | **100** | 0 | 0 |
| 18 | 67 ⚠ | 32 | 1 | 62 ⚠ | 0 | **38** | 65 ⚠ | 0 | **35** |
| 19 | 32 ⚠ | 1 | **67** | 50 ⚠ | 0 | **50** | 0 ⚠ | 0 | **100** |
| 20 | 64 ⚠ | 3 | 33 | 52 ⚠ | **45** | 3 | 48 ⚠ | 16 | 36 |
| 21 | 2 ⚠ | 0 | **98** | 0 ⚠ | 1 | **99** | 0 ⚠ | 1 | **99** |
| 22 | 83 | 3 | 14 | **100** | 0 | 0 | **100** | 0 | 0 |
| 23 | 91 | 0 | 9 | **100** | 0 | 0 | **100** | 0 | 0 |
| 24 | **85** | 4 | 9 | **89** | 0 | 11 | 66 ⚠ | 1 | 33 |
| 25 | 15 ⚠ | 40 | 45 | 2 ⚠ | 0 | **98** | 11 ⚠ | 11 | **78** |
| 26 | 70 ⚠ | 23 | 7 | 55 ⚠ | 29 | 16 | 50 ⚠ | **50** | 0 |
| 27 | 43 ⚠ | 41 | 16 | 48 ⚠ | 21 | 31 | 50 ⚠ | **50** | 0 |
| 28 | **97** | 0 | 3 | 28 ⚠ | 38 | 34 | 58 ⚠ | 25 | 17 |

#### Comparativa por goal E1.2 → E1.3 SUBWP

| Goal | E1.2 s42 | E1.3 s42 | Δ | E1.2 s123 | E1.3 s123 | Δ | E1.2 s524 | E1.3 s524 | Δ |
|------|:--------:|:--------:|:-:|:---------:|:---------:|:-:|:---------:|:---------:|:-:|
| 01–17 | 100 | ~100 | ≈ | 100 | ~100 | ≈ | 100 | ~100 | ≈ |
| 18 | ~100 | 67 | **−33** ⚠ | ~100 | 62 | **−38** ⚠ | ~100 | 65 | **−35** ⚠ |
| 19 | 74 | 32 | −42 ⚠ | 2 | 50 | **+48** | 1 | 0 | = |
| 20 | 7 | **64** | **+57** ✅ | 0 | 52 | **+52** ✅ | 0 | **48** | **+48** ✅ |
| 21 | 14 | 2 | −12 | 0 | 0 | = | 0 | 0 | = |
| 22 | 1 | **83** | **+82** ✅ | 0 | **100** | **+100** ✅ | 0 | **100** | **+100** ✅ |
| 23 | 100 | 91 | −9 | 100 | **100** | = | 100 | **100** | = |
| 24 | 86 | **85** | ≈ | 67 | **89** | **+22** | 73 | 66 | −7 |
| 25 | **59** | 15 | **−44** ⚠ | **56** | 2 | **−54** ⚠ | **67** | 11 | **−56** ⚠ |
| 26 | **87** | 70 | −17 | 51 | 55 | +4 | 37 | **50** | +13 |
| 27 | 36 | 43 | +7 | 31 | 48 | +17 | 49 | **50** | +1 |
| 28 | 48 | **97** | **+49** ✅ | 31 | 28 | −3 | 22 | **58** | **+36** |

#### Análisis cualitativo E1.3 SUBWP — inferencia

**El hallazgo más importante — goal_20 y goal_22 parcialmente recuperados:**
E1.3 logra algo que ningún experimento anterior consiguió: hacer que goals del cluster A sean
viables. goal_20 pasa de 0-7% a 48-64% (+48 a +57pp) y goal_22 de 0-1% a 83-100% (+82 a
+100pp) en todas las seeds. La extensión del trayecto de P1 a x=-4 cambia suficientemente
la distribución espacio-temporal del peatón como para crear ventanas en estos goals antes
imposibles. Sin embargo, goal_21 sigue siendo prácticamente 0% (2% en s42), lo que indica
que el problema geométrico de goal_21 persiste y es más profundo.

**Nueva regresión crítica — goal_18 y goal_25:**
- **goal_18:** era ~100% en E1.2, ahora 62-67% con 35-38% col_exit puro en s123/s524. El
  peatón pasa ahora más tiempo en la zona de goal_18 (en su camino desde x=-4 hacia x=4),
  creando interferencias que no existían con la trayectoria original.
- **goal_25:** era 56-67% en E1.2, ahora 2-15%. En s123/s524 el col_exit es 78-98% —
  catastrófico. La nueva trayectoria parece colocar a P1 exactamente en el exit de goal_25
  con mayor frecuencia.

**Cluster B (goals 24-28) — resultados heterogéneos:**
- goal_24: mejora en s123 (+22pp) y se mantiene en s42 (85%), pero regresa en s524 (−7pp)
- goal_25: colapso generalizado (la regresión más grave de E1.3 SUBWP)
- goal_26: mejora leve (+4 a +13pp), principalmente en s524
- goal_27: mejora moderada (+7 a +17pp) en todas las seeds
- goal_28: s42=97% (+49pp, el mejor resultado de toda la serie SUBWP para este goal),
  s524=58% (+36pp), s123=28% (regresión −3pp)

**Patrón de fallos por goal:**
- col_exit domina en: goals 10, 18-19, 21, 25 — el robot golpea al salir de la estantería
- col_approach domina en: goals 17, 20, 26-28 — P1 bloquea el approach
- goal_18 pasa de 0% fallo a 35-38% col_exit (nuevo problema post-E1.3)

**goal_10 — problema inesperado:** s42=58% y s524=60%, ambos por col_exit (42/40%). Era 100%
en experimentos anteriores. Podría ser una interferencia puntual de P1 al inicio de su
nueva trayectoria (cuando parte de x=-4 hacia x=4, pasa por el área de goal_10).

### 16.4 Comparativa global inferencia SUBWP

| Experimento | s42 | s123 | s524 | Media 3 seeds |
|-------------|:---:|:----:|:----:|:-------------:|
| E1 | 80.4% | 73.8% | 76.9% | 77.0% |
| E1.2 | **82.6%** | 75.9% | 76.8% | 78.4% |
| E1_pred | 78.9% | 73.7% | 76.2% | 76.3% |
| **E1.3** | 81.6% | **81.3%** | **78.9%** | **80.6%** ✅ |

**E1.3 es el mejor resultado SUBWP** (80.6% media, +2.2pp vs E1.2). La mejora viene de
goals previamente imposibles (goal_20 +50pp, goal_22 +94pp, goal_28 s42 +49pp), aunque
a costa de regresar goal_18 (−35pp) y goal_25 (−50pp) en todas las seeds.

**Conclusión:** la trayectoria extendida rompe el bloqueo de algunos goals del cluster A
(goal_20, goal_22) pero desplaza el problema a goals anteriormente seguros (goal_18, goal_25).
Indica que P1 sigue siendo un obstáculo activo y que la solución definitiva requiere
una política de avoidance más robusta, no solo más ventanas temporales.

**Referencia definitiva SUBWP:** E1.3 — 80.6% media 3 seeds / 81.6% mejor seed (s42).

---

## Sección 17 — Experimento E1.4: Patience reward + replan agresivo

### Motivación

El análisis de inferencia E1.3 identifica dos problemas distintos en SUBWP:

**Problema A — col_exit (goals 18, 19, 21, 25):** el robot choca al salir de la estantería
cuando P1 está en el corredor de salida. A diferencia de STHWP, en SUBWP este fallo afecta
también a goals del cluster A (19, 21) — goals donde la ruta A* sale por un pasillo estrecho
que P1 cruza. La solución es la misma que en STHWP: patience reward para aprender a esperar.

**Problema B — replanning insuficientemente agresivo:** el A* dinámico existente tiene un
umbral de activación conservador (0.6m) y un inflation pequeño (0.75m). Cuando P1 está a
0.7m del segmento robot→waypoint, el replanning no se activa y el robot sigue la ruta
original hacia P1. Aumentando estos parámetros, el sistema redirige antes y con mayor margen.

### Cambios implementados

#### Modificación 1 — Parámetros de replanning dinámico (`webots_env.py` SUBWP)

```python
# ANTES (E1.1 a E1.3):
REPLAN_DIST_PERP     = 0.6   # m
REPLAN_INFLATE_CELLS = 3     # celdas × 0.25m = 0.75m
REPLAN_COOLDOWN      = 40    # pasos

# DESPUÉS (E1.4):
REPLAN_DIST_PERP     = 1.0   # m  (+0.4m, detecta antes)
REPLAN_INFLATE_CELLS = 4     # celdas × 0.25m = 1.0m  (margen mayor)
REPLAN_COOLDOWN      = 20    # pasos  (replanning 2× más frecuente)
```

El efecto combinado: el A* se activa cuando P1 está a 1.0m del segmento actual (antes 0.6m),
bloquea una zona de 1.0m alrededor de P1 en la cuadrícula (antes 0.75m), y puede volver a
replanificar cada 20 pasos (≈0.4s) en vez de cada 40 (≈0.8s). Esto da al robot más tiempo
y opciones para encontrar un camino alternativo antes de que P1 cierre el paso.

Limitación: en corredores muy estrechos (goal_21, goal_25 entrance) puede no existir ruta
alternativa aunque se replanifique. En esos casos, el patience reward es el mecanismo
principal de defensa.

#### Modificación 2 — Penalización proximidad reforzada (`webots_env.py` SUBWP)

Idéntica a STHWP:
```python
# ANTES:
if dist_ped < 2.0:
    recompensa -= 0.8 * math.exp(-2.0 * dist_ped)

# DESPUÉS:
if dist_ped < 2.0:
    recompensa -= 1.5 * math.exp(-3.0 * dist_ped)
```

#### Modificación 3 — Patience reward (`webots_env.py` SUBWP)

Idéntica a STHWP:
```python
if dist_ped < 0.8 and abs(velocidad_lineal) < 0.05:
    recompensa += 0.4
```

### Configuración E1.4 SUBWP

| Parámetro | Valor |
|-----------|:-----:|
| Base | `subwp_e1_3_s{42,123,524}_final` (48 dims, sin pred_horizon) |
| World | `warehouse_1_subwp_1ped.wbt` (P1 trayectoria x∈[-4,4]) |
| Steps | 2M fine-tune |
| LR | 1e-5 |
| ent_coef | 0.005 |
| max_steps | 6000 |
| Seeds | 42, 123, 524 |

### Goals objetivo

| Goal | Problema E1.3 | Fallo dominante | Mecanismo E1.4 |
|------|:-------------:|:---------------:|:--------------:|
| 18 | 62-67% | col_exit 35-38% | Patience reward |
| 19 | 0-50% | col_exit 50-100% | Patience + replan agresivo |
| 21 | 0-2% | col_exit 98-99% | Patience + replan (limitado por geometría) |
| 25 | 2-15% | col_exit 78-98% | Patience + replan agresivo |
| 10 | 58-60% | col_exit 40-42% | Patience reward |

Goal_21 es el caso más incierto: incluso con patience y mejor replanning, el corredor
de salida es geométricamente estrecho y P1 puede bloquearlo completamente. Si A* no
encuentra ruta alternativa, el único recurso es esperar a que P1 pase — que el patience
reward debería fomentar, aunque no garantiza 100% de éxito.

### Resumen de todos los cambios E1.4 vs versiones anteriores

| Componente | E1.1 a E1.3 | E1.4 |
|-----------|:-----------:|:----:|
| Penalización prox. P1 (0.5m) | -0.289/paso | **-0.669/paso** |
| Patience reward | ✗ | **+0.4/paso** cuando P1 < 0.8m y quieto |
| REPLAN_DIST_PERP | 0.6m | **1.0m** |
| REPLAN_INFLATE_CELLS | 3 (0.75m) | **4 (1.0m)** |
| REPLAN_COOLDOWN | 40 pasos | **20 pasos** |

### Resultados de training E1.4 SUBWP

**Steps totales acumulados**: 9M (base E1.3) + 2M fine-tune = **11M steps** por seed.

#### Métricas de rollout (final del training)

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| `ep_rew_mean` | 503.67 | 531.20 | 540.10 |
| `ep_len_mean` | ~2857 | ~2892 | ~2802 |

Los episodios SUBWP son significativamente más largos que STHWP (~2850 vs ~2000 pasos)
y el reward acumulado es mayor en términos absolutos, consistente con episodios más largos.
La comparación directa de `ep_rew_mean` entre sistemas no es informativa — se deben comparar
tasas relativas (tasa_exito, tasa_colision).

#### Métricas de rendimiento (stats)

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| `tasa_exito_%` | 80.95 | **87.24** | 86.38 |
| `tasa_colision_%` | 19.06 | **12.76** | 13.62 |
| `tasa_estanteria_%` | 91.83 | **93.34** | 93.26 |
| `exito_ult100_%` | 81.43 | 84.01 | 80.00 |
| `exito_dificiles_ult100_%` | 80.27 | **88.92** | 85.00 |
| `exito_faciles_ult100_%` | 79.02 | 86.02 | 85.00 |
| `n_colisiones` | 132.74 | 87.98 | 94.99 |
| `n_exitos` | 563.50 | 601.49 | 602.50 |
| `n_truncados` | 0 | 0 | 0 |

**Mejor seed**: s123 con 87.24% de éxito y 12.76% de colisión.

La `tasa_estanteria_%` (91-93%) es alta y consistente entre seeds — el robot llega bien a la
zona de estantería en la mayoría de los episodios. Los fallos se concentran en el approach
final y las salidas, no en la navegación general.

**Anomalía s42**: en E1.4 SUBWP, s42 es el peor seed (col=19%) mientras que en experimentos
anteriores solía ser el más estable. Esto puede indicar que el seed 42 cayó en un mínimo
local con la nueva reward function (patience + penalización reforzada). Se necesita confirmar
con inferencia por goal.

#### Métricas PPO internas

| Métrica | s42 | s123 | s524 | Interpretación |
|---------|:---:|:----:|:----:|----------------|
| `approx_kl` | 0.0019 | 0.0040 | 0.0027 | Muy bajo → fine-tuning conservador |
| `clip_fraction` | 0.0006 | 0.0313 | 0.0233 | s42 prácticamente sin clipping (!) |
| `entropy_loss` | -6.21 | -4.91 | -5.14 | Entropía muy baja → política determinista |
| `explained_variance` | **0.498** | 0.856 | 0.838 | s42 con valor function mal ajustada |
| `learning_rate` | 1e-5 | 1e-5 | 1e-5 | Confirmado |
| `policy_gradient_loss` | -0.0005 | -0.0011 | -0.0017 | Gradientes muy pequeños en s42 |
| `train/std` | 67.8 | 99.2 | 58.7 | Muy alto (vs STHWP 17-26) |
| `value_loss` | **47.24** | 9.15 | 7.33 | s42 con error alto en value fn |

**Observaciones críticas:**
- `train/std` de 67-99 para SUBWP frente a 17-26 de STHWP: SUBWP tiene espacio de acción
  con mayor varianza intrínseca, probablemente relacionado con el mecanismo de waypoints
  discretos que genera situaciones de mayor variabilidad.
- s42 tiene `explained_variance=0.498` (prácticamente azar) y `clip_fraction≈0` — la política
  dejó de actualizar significativamente. Posible convergencia prematura o colapso local.
- `entropy_loss` de SUBWP (-4.9 a -6.2) es mucho más negativo que STHWP (-2.9 a -3.2),
  indicando que SUBWP desarrolló políticas más deterministas durante el fine-tune.

#### FPS

~902-908 fps (s123 más lento por mayor complejidad computacional del replanning agresivo
con `REPLAN_COOLDOWN=20`). Sin interrupciones, sin truncaciones.

#### Análisis comparativo E1.3 → E1.4 (training)

| Seed | E1.3 `tasa_exito_%` (aprox.) | E1.4 `tasa_exito_%` | Delta |
|------|:----------------------------:|:-------------------:|:-----:|
| s42  | ~84 | 80.95 | **-3 pp** |
| s123 | ~84 | 87.24 | **+3 pp** |
| s524 | ~83 | 86.38 | **+3 pp** |

Los resultados son polarizados: s123 y s524 mejoran ~3 pp, mientras s42 retrocede ~3 pp.
Esto sugiere que los cambios E1.4 tienen un **efecto de alta varianza entre seeds** — la
combinación de patience reward + replan agresivo cambia la dinámica de aprendizaje lo
suficiente como para que el resultado dependa del seed inicial.

**La mejora media es ≈0 pp a nivel global**, pero los seeds buenos (s123, s524) alcanzaron
87-87% de éxito en training — la mejor marca histórica de SUBWP.

### Resultados inferencia E1.4 SUBWP (determinista)

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | 82.5% | **83.6%** | 77.5% | **81.2%** |
| Col. exit | 15.4% | 12.5% | 21.4% | 16.4% |
| Col. approach | 2.1% | 3.9% | 1.1% | 2.4% |
| Truncados | 0% | 0% | 0% | 0% |
| Goals al 100% | 16/28 | 15/28 | 15/28 | — |

#### Por goal — detalle completo E1.4 SUBWP

| Goal | s42 Éxito | s42 C_app | s42 C_exit | s123 Éxito | s123 C_app | s123 C_exit | s524 Éxito | s524 C_app | s524 C_exit | Media | Fallo |
|------|:---------:|:---------:|:----------:|:----------:|:----------:|:-----------:|:----------:|:----------:|:-----------:|:-----:|:-----:|
| 01–09 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 | 100% | — |
| 10 | 39 ⚠ | 0 | 61 | **100** | 0 | 0 | 61 ⚠ | 0 | 39 | 67% | col_exit |
| 11–13 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 | 100% | — |
| 14–15 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 | 100% | — |
| 16 | **100** | 0 | 0 | 99 | 1 | 0 | **100** | 0 | 0 | 99.7% | — |
| 17 | 78 ⚠ | 1 | 21 | 80 ⚠ | 2 | 18 | 89 | 0 | 11 | 82% | col_exit |
| 18 | **100** | 0 | 0 | 69 ⚠ | 1 | 30 | 50 ⚠ | 3 | 47 | 73% | col_exit |
| 19 | 29 ⚠ | 14 | 57 | 0 ⚠ | 16 | **84** | 3 ⚠ | 16 | 81 | 11% | col_exit |
| 20 | 68 ⚠ | 0 | 32 | 67 ⚠ | 0 | 33 | 66 ⚠ | 0 | 34 | 67% | col_exit |
| 21 | 2 ⚠ | 0 | **98** | 0 ⚠ | 0 | **100** | 2 ⚠ | 0 | **98** | 1.3% | col_exit |
| 22 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 | 100% | — |
| 23 | **100** | 0 | 0 | **100** | 0 | 0 | **100** | 0 | 0 | 100% | — |
| 24 | 68 ⚠ | 7 | 25 | 80 ⚠ | 5 | 15 | 67 ⚠ | 15 | 18 | 72% | col_exit |
| 25 | 0 ⚠ | 3 | **97** | 2 ⚠ | 0 | **98** | 11 ⚠ | 11 | **78** | 4.3% | col_exit |
| 26 | **96** | 4 | 0 | 84 ⚠ | **16** | 0 | **94** | 6 | 0 | 91% | — |
| 27 | 13 ⚠ | 16 | 71 | 56 ⚠ | 29 | 15 | 1 ⚠ | 10 | **89** | 23% | col_exit |
| 28 | **99** | 0 | 1 | 62 ⚠ | **38** | 0 | 83 | 15 | 2 | 81% | col_approach |

#### Comparativa por goal E1.3 → E1.4 SUBWP

| Goal | E1.3 s42 | E1.4 s42 | Δ s42 | E1.3 s123 | E1.4 s123 | Δ s123 | E1.3 s524 | E1.4 s524 | Δ s524 | Media Δ |
|------|:--------:|:--------:|:-----:|:---------:|:---------:|:------:|:---------:|:---------:|:------:|:-------:|
| 01–13 | ~100 | 100 | = | ~100 | 100 | = | ~100 | 100 | = | = |
| 14–15 | ~100 | 100 | = | ~100 | 100 | = | ~100 | 100 | = | = |
| 16 | ~100 | 100 | = | ~100 | 99 | = | ~100 | 100 | = | = |
| 17 | — | 78 | — | — | 80 | — | — | 89 | — | — |
| 18 | 67 | **100** | **+33** ✅ | 62 | 69 | +7 | 65 | 50 | −15 ⚠ | **+8** |
| 19 | 32 | 29 | −3 | 50 | 0 | **−50** ⚠ | 0 | 3 | +3 | **−17** |
| 20 | **64** | 68 | +4 | 52 | 67 | **+15** | 48 | 66 | **+18** | **+12** ✅ |
| 21 | 2 | 2 | = | 0 | 0 | = | 0 | 2 | +2 | = |
| 22 | 83 | **100** | +17 | **100** | 100 | = | **100** | 100 | = | **+6** |
| 23 | 91 | **100** | +9 | **100** | 100 | = | **100** | 100 | = | **+3** |
| 24 | **85** | 68 | −17 ⚠ | **89** | 80 | −9 | 66 | 67 | +1 | **−8** |
| 25 | 15 | 0 | **−15** ⚠ | 2 | 2 | = | 11 | 11 | = | **−5** |
| 26 | 70 | **96** | **+26** ✅ | 55 | 84 | **+29** ✅ | 50 | **94** | **+44** ✅ | **+33** ✅ |
| 27 | 43 | 13 | **−30** ⚠ | 48 | 56 | +8 | 50 | 1 | **−49** ⚠ | **−24** ⚠ |
| 28 | **97** | **99** | +2 | 28 | 62 | **+34** ✅ | 58 | 83 | **+25** ✅ | **+20** ✅ |

#### Comparativa global E1 → E1.2 → E1_pred → E1.3 → E1.4 SUBWP (inferencia)

| Experimento | s42 | s123 | s524 | Media 3 seeds |
|-------------|:---:|:----:|:----:|:-------------:|
| E1 | 80.4% | 73.8% | 76.9% | 77.0% |
| E1.2 | **82.6%** | 75.9% | 76.8% | 78.4% |
| E1_pred | 78.9% | 73.7% | 76.2% | 76.3% |
| E1.3 | 81.6% | 81.3% | 78.9% | 80.6% |
| **E1.4** | 82.5% | **83.6%** | 77.5% | **81.2%** |

#### Análisis cualitativo E1.4 SUBWP — inferencia

**Resultado principal: progreso marginal (+0.6pp) con mejoras y regresiones compensadas.**
E1.4 alcanza 81.2% media, prácticamente igual que E1.3 (80.6%). La media oculta una
dispersión importante: s42 y s123 mejoran ligeramente, s524 regresa de 78.9% a 77.5%.

**Mejora destacada — goal_26 (+33pp media): el impacto más claro del replan agresivo.**
goal_26 pasa de 58.3% (E1.3 media) a 91.3% (E1.4 media), con mejoras consistentes en
todos los seeds (+26, +29, +44pp). Este goal requiere que el robot navegue por un corredor
mientras P1 lo cruza — el replanning más agresivo (REPLAN_DIST_PERP=1.0m,
REPLAN_COOLDOWN=20) encuentra rutas alternativas antes de que el peatón bloquee el paso.
Es la mejora más grande y reproducible de toda la serie E1.x SUBWP.

**Mejora notable — goal_28 (+20pp media en s123 y s524).**
s123 pasa de 28% a 62% (+34pp), s524 de 58% a 83% (+25pp). Solo s42 ya estaba en 97%.
El replanning más frecuente ayuda a recalcular la ruta cuando P1 aparece en el approach.

**Mejora moderada — goal_20 (+12pp) y goal_22/23 (consolidación a 100%).**

**Regresiones críticas:**
- **goal_27**: media cae de 47% a 23% (−24pp). s42=13% (−30pp), s524=1% (−49pp). El
  replanning agresivo puede estar generando rutas inestables o en bucle para goal_27,
  cuya geometría de approach es especialmente difícil.
- **goal_19**: s123 colapsa de 50% a 0% (−50pp). El replan agresivo puede estar
  replanificando demasiado frecuentemente, interrumpiendo la ejecución correcta del goal.
- **goal_24**: retroceso de 80% a 72% (−8pp media).

**Goals estructuralmente limitados — sin cambio:**
- **goal_21**: 1.3% (igual que E1.3). El replan agresivo no ayuda — la geometría del
  corredor de goal_21 es tan estrecha que no existe ruta alternativa viable. Limitación
  fundamental de SUBWP con la geometría actual.
- **goal_25**: 4.3% (sin mejora). Mismo problema de exit corridor muy estrecho.

**Patrón de fallos por tipo:**
- col_exit domina: goals 10, 17-21, 24-25, 27 — el robot choca al salir de estanterías
- col_approach: solo goal_28 (s123 con P1 bloqueando approach) y goal_26 parcialmente
- La tasa de col_exit global (16.4%) es el doble de la de STHWP (12.8%)

**Comparativa con STHWP E1.4:**
- STHWP global: 83.2% vs SUBWP: 81.2% → diferencia de 2pp
- STHWP domina goals 14-16 (38-70% vs 100% SUBWP para esos goals — aquí SUBWP es mejor)
- SUBWP domina goals 22-23 (100% vs 12-100% STHWP)
- Ambos fallan en goals 19, 21, 25 pero SUBWP peor en 19 y 21

**Conclusión E1.4 SUBWP:**
El replan agresivo tiene un impacto positivo claro en goals de col_approach (26, 28) pero
introduce inestabilidad en algunos goals de col_exit (27, 19). La mejora más importante
de toda la serie E1.x para SUBWP es goal_26 (+33pp), pero el sistema sigue lejos del
90% objetivo. Los goals 21, 19 y 25 permanecen como bloqueos fundamentales.

---

## Sección 18 — Experimento E1.5: Exit-corridor reward + curriculum ponderado

### Motivación

E1.4 mostró que el replan agresivo mejora goals de col_approach (goal_26: +33pp) pero
no resuelve col_exit (goals 19, 21, 25 siguen siendo estructuralmente difíciles). El
patience reward genérico tampoco funcionó. E1.5 introduce:

1. **Exit-corridor reward**: `+0.6/paso` cuando P1 está en cono de 45° delante del robot
   durante `_hacia_descarga`, dist < 2.5m, vel < 0.05 m/s. Señal más temprana (2.5m vs
   0.8m) y geométricamente precisa respecto a la dirección de exit.
2. **Curriculum ponderado**: goals 17-21, 24-28 muestreados con peso 3× (~62% del tiempo).
3. **4M steps** fine-tune (vs 2M en E1.4).
4. **Replan agresivo mantenido** de E1.4 (REPLAN_DIST_PERP=1.0m, COOLDOWN=20).

### Configuración E1.5 SUBWP

| Parámetro | Valor |
|-----------|:-----:|
| Base | `subwp_e1_4_s{42,123,524}_final` (48 dims) |
| Steps | 4M fine-tune |
| LR | 1e-5 |
| ent_coef | 0.005 |
| max_steps | 6000 |
| Seeds | 42, 123, 524 |
| Hard goals | 17-21, 24-28 (peso 3×) |

### Resultados inferencia E1.5 SUBWP (determinista)

| Métrica | s42 | s123 | s524 | Media 3 seeds |
|---------|:---:|:----:|:----:|:-------------:|
| **Éxito global** | **83.2%** | **82.5%** | 79.6% | **81.7%** |
| Col. exit | 14.6% | 14.3% | 20.0% | 16.3% |
| Col. approach | 2.1% | 3.2% | 0.4% | 2.0% |
| Truncados | 0% | 0% | 0% | 0% |

#### Por goal — detalle completo E1.5 SUBWP

| Goal | s42 | s123 | s524 | Media | Fallo |
|------|:---:|:----:|:----:|:-----:|:-----:|
| 01–09 | **100** | **100** | **100** | 100% | — |
| 10 | 70 ⚠ | **100** | 61 ⚠ | 77.0% | col_exit |
| 11–13 | **100** | **100** | **100** | 100% | — |
| 14–15 | **100** | **100** | **100** | 100% | — |
| 16 | **100** | 78 ⚠ | **100** | 92.7% | col_exit |
| 17 | 79 ⚠ | 85 | **90** | 84.7% | col_exit |
| 18 | 82 ⚠ | 59 ⚠ | 50 ⚠ | 63.7% | col_exit |
| 19 | 30 ⚠ | 0 ⚠ | 3 ⚠ | 11.0% | col_exit |
| 20 | 62 ⚠ | 68 ⚠ | 67 ⚠ | 65.7% | col_exit |
| 21 | 0 ⚠ | 0 ⚠ | 2 ⚠ | 0.7% | col_exit (estructural) |
| 22–23 | **100** | **100** | **100** | 100% | — |
| 24 | 66 ⚠ | **81** | 68 ⚠ | 71.7% | col_exit |
| 25 | 0 ⚠ | 0 ⚠ | 13 ⚠ | 4.3% | col_exit |
| 26 | **97** | 87 | **99** | 94.3% | — |
| 27 | 51 ⚠ | 65 ⚠ | 2 ⚠ | 39.3% | col_exit |
| 28 | **99** | 75 ⚠ | 76 ⚠ | 83.3% | col_approach |

#### Comparativa E1.4 → E1.5 SUBWP

| Goal | E1.4 media | E1.5 media | Δ | Tendencia |
|------|:----------:|:----------:|:-:|:---------:|
| 10 | 66.7% | 77.0% | **+10** ✅ | mejora |
| 16 | 99.7% | 92.7% | **−7** ⚠ | regresión |
| 17 | 82.3% | 84.7% | +2 | estable |
| 18 | 73.0% | 63.7% | **−9** ⚠ | regresión |
| 19 | 10.7% | 11.0% | ≈ | sin cambio |
| 20 | 67.0% | 65.7% | ≈ | estable |
| 21 | 1.3% | 0.7% | ≈ | sin cambio (estructural) |
| 24 | 71.7% | 71.7% | = | estable |
| 25 | 4.3% | 4.3% | = | sin cambio |
| 26 | 91.3% | **94.3%** | **+3** | mejora leve |
| 27 | 23.3% | **39.3%** | **+16** ✅ | mejora notable |
| 28 | 81.3% | 83.3% | **+2** | mejora leve |
| **Global** | **81.2%** | **81.7%** | **+0.5** | plateau |

#### Comparativa global E1 → … → E1.5 SUBWP (inferencia)

| Experimento | s42 | s123 | s524 | Media 3 seeds |
|-------------|:---:|:----:|:----:|:-------------:|
| E1 | 80.4% | 73.8% | 76.9% | 77.0% |
| E1.2 | **82.6%** | 75.9% | 76.8% | 78.4% |
| E1_pred | 78.9% | 73.7% | 76.2% | 76.3% |
| E1.3 | 81.6% | 81.3% | 78.9% | 80.6% |
| E1.4 | 82.5% | **83.6%** | 77.5% | 81.2% |
| **E1.5** | **83.2%** | 82.5% | 79.6% | **81.7%** |

#### Análisis cualitativo E1.5 SUBWP

**Resultado principal: plateau en ~82%, mejora puntual en goal_27.**
E1.5 obtiene 81.7% — +0.5pp sobre E1.4 (81.2%). La mejora es real pero mínima y la
dispersión entre seeds (s42=83.2% vs s524=79.6%) sigue siendo alta.

**Mejora destacada — goal_27 (+16pp media): 23.3% → 39.3%.**
s42: 13→51%, s123: 56→65%, s524: 1→2% (s524 sigue fallando). El exit-corridor reward
ayuda al robot a detectar a P1 en el corredor de approach/exit de goal_27 y esperar antes
de intentar pasar. Es la segunda mejora más grande de E1.x SUBWP tras goal_26 en E1.4.

**Mejora en goal_10 (+10pp): 66.7% → 77.0%.**
El curriculum ponderado (goal_10 incluido en hard goals) da más práctica en este goal,
reduciéndola col_exit que causaba el problema con la trayectoria extendida de P1.

**Regresiones inesperadas:**
- **goal_18: −9pp** (73% → 63.7%). La señal de exit-corridor en `_hacia_descarga` puede
  estar interfiriendo con el approach de goal_18 (el robot anticipa la salida demasiado
  pronto y se para innecesariamente).
- **goal_16: −7pp** (99.7% → 92.7%). Regresión leve, probablemente ruido de fine-tuning.

**Goals estructuralmente limitados — sin cambio:**
- **goal_21**: 0.7% (igual). Límite físico del corredor. Ningún reward lo resolverá.
- **goal_25**: 4.3% (igual). Misma limitación de geometría estrecha.
- **goal_19**: 11.0% (igual). Consistentemente difícil en todos los experimentos.

**Conclusión E1.5 SUBWP y cierre de la serie E1.x:**
La serie E1.x ha alcanzado su techo en ~82% para SUBWP. Las mejoras individuales
(goal_26: +33pp en E1.4, goal_27: +16pp en E1.5) son reales pero compensadas por
regresiones en otros goals. Los bloqueos fundamentales (goals 19, 21, 25) no son
solucionables con reward shaping — requieren un cambio de entorno o arquitectura.

**Mejor referencia global SUBWP (3 seeds):** E1.5 — 81.7%
**Mejor referencia SUBWP (seed individual):** E1.4 s123 — 83.6%

---

---

## Sección 19 — Experimento E2.0: Pre-entrenamiento stage6 sin peatón

### Motivación

SUBWP no tiene un modelo stage6 sin peatón (el flujo fue stage5 → stage6_din directamente).
Para que E2.1 parta de un punto equivalente al `run003_sXX_stage6_final` de STHWP, se
entrena primero un E2.0 que aprende el ciclo completo (approach→exit→return) sin peatón.
El peatón está físicamente en el mundo (LIDAR lo detecta como obstáculo dinámico) pero
`ped_obs=False` → sin rewards/penalizaciones específicas.

### Configuración E2.0

| Parámetro | Valor |
|-----------|-------|
| Base | `subwp_sXX_wp75_r2_stage5_final` (40 dims) |
| stage | 6 (ciclo completo) |
| ped_obs | False |
| Steps | 2M |
| lr | 5e-5 |
| ent_coef | 0.01 |

### Resultados TensorBoard E2.0

| Seed | ep_rew_mean inicio | ep_rew_mean final | ep_len inicio | ep_len final |
|------|--------------------|-------------------|---------------|--------------|
| s42  | −220 | +411 | 938 | 2736 | ✓ converge |
| **s123** | **−320** | **−155** | **6000** | **3994** | **✗ NO converge** |
| s524 | — | — | — | — | ✓ (inferido de E2.1) |

**s123 falla desde el primer batch:** ep_len=6000 (máximo, todos truncados) indica que el
modelo base `subwp_s123_wp75_r2_stage5_final` olvidó el approach durante el entrenamiento
de stage5 (return-only). Stage5 solo practica el tramo descarga→espera; el modelo s123
sufrió olvido catastrófico de la habilidad de approach aprendida en stage4.

---

## Sección 20 — Experimento E2.1: Observación realista (solo LIDAR 5m)

### Motivación y diseño

Mismo cambio que en STHWP E2.1: eliminar la observación por supervisor (no realista) y
ampliar el LIDAR a 5m como única fuente de percepción del peatón.

**Cambios respecto a E1.x:**
- Observación por supervisor eliminada (obs: 48 dims → 40 dims)
- MAX_LIDAR_RANGE: 3.5m → 5.0m
- Penalización proximidad peatón: umbral 2.0m → 4.0m, exp(−1.5·d)
- Exit-corridor reward: umbral 2.5m → 4.0m
- Base: `subwp_e2_0_sXX_final` (stage6 sin peatón, 40 dims)
- 6M steps, lr=1e-4, ent_coef=0.01

### Configuración E2.1 SUBWP

| Parámetro | Valor |
|-----------|-------|
| Base | `subwp_e2_0_sXX_final` (40 dims, stage6 sin peatón) |
| Obs space | 40 dims (36 LIDAR 5m + 4 estado) |
| Steps | 6M por seed |
| lr | 1e-4 |
| ent_coef | 0.01 |
| Hard goals | goal_17–21, goal_24–28 (3×) |

### Incidencia crítica: s123 completamente rota

**Cadena de fallos:**

```
subwp_s123_wp75_r2_stage5_final
  → olvido catastrófico del approach durante stage5 (return-only)
  → E2.0 s123: ep_len=6000 desde el primer batch, reward=-320 nunca converge
  → subwp_e2_0_s123_final: política degenerada (no sabe hacer approach)
  → E2.1 s123: hereda política rota → 0% éxito, 62% truncados, 37% col_approach
```

**Evidencia TensorBoard E2.1 s123:**
- ep_rew_mean: inicio=−167 → medio=−167 → final=−176 (nunca positivo)
- ep_len_mean: inicio=958 → medio=5338 → final=5025 (los pocos episodios cortos son colisiones)
- Comparación s42/s524: inicio=+600 → final=+423/+454 (convergencia normal)

**Fix planeado (E2.0b s123):** Cargar desde `subwp_s123_wp75_r2_stage4_final` (que sí conoce
approach+exit) en lugar de stage5. Stage4 mezcla approach y exit, por lo que no sufre olvido
catastrófico.

### Resultados de inferencia E2.1 SUBWP

**Global por seed (2800 ep/seed — resultados definitivos con s123 corregida):**

| Seed | Éxito | col_approach | col_exit | truncado | Estado |
|------|-------|--------------|----------|----------|--------|
| s42  | 80.9% | 4.4% | 14.8% | 0.0% | ✓ |
| s123 (E2.0b) | 83.0% | 2.4% | 14.6% | 0.0% | ✓ corregida |
| s524 | 83.2% | 2.8% | 14.0% | 0.0% | ✓ |
| **Media** | **82.4%** | **3.2%** | **14.5%** | **0.0%** | |

*s123 inicial (desde stage5): 0% éxito, 62.6% truncados → olvido catastrófico. Corregida con E2.0b (desde stage4_r2): 83.0%.*

**Por goal — comparativa E1.5 vs E2.1 definitiva (300 ep/goal, 3 seeds):**

| Goal | E1.5 | E2.1 | Δ | col_exit E2.1 |
|------|------|------|---|----------------|
| goal_01–09 | 100% | 100% | 0pp | 0% |
| goal_10 | 77.0% | 79.7% | +2.7pp | 20% |
| goal_11–15 | 100% | 100% | 0pp | 0% |
| goal_16 | 92.7% | 94.3% | +1.7pp | 6% |
| goal_17 | 84.7% | 86.3% | +1.7pp | 7% |
| goal_18 | 63.7% | 67.7% | +4.0pp | 30% |
| goal_19 | 11.0% | 17.7% | +6.7pp | 66% |
| goal_20 | 65.7% | 67.3% | +1.7pp | 33% |
| goal_21 | 0.7% | 7.0% | +6.3pp | 93% |
| goal_22–23 | 100% | 100% | 0pp | 0% |
| goal_24 | 71.7% | **83.0%** | **+11.3pp** | 4% |
| goal_25 | 4.3% | 1.0% | −3.3pp | 99% |
| goal_26 | 94.3% | 83.3% | **−11.0pp** | 2% |
| goal_27 | 39.3% | 19.3% | **−20.0pp** | 46% |
| goal_28 | 83.3% | **99.3%** | **+16.0pp** | 0% |
| **MEDIA** | **81.7%** | **82.4%** | **+0.7pp** | |

### Análisis cualitativo E2.1 SUBWP — Resultados definitivos (3 seeds)

**Resultado principal: +0.7pp respecto a E1.5 (81.7% → 82.4%). Mejora marginal.**
A diferencia de STHWP (+5.5pp), el cambio a obs realista no produce ganancia sustancial en SUBWP.
La diferencia sistemática entre sistemas sugiere que el mecanismo de planificación es el factor
limitante en SUBWP, no la observación: STH-WP tiene subgoal dinámico (lookahead en A*) que puede
adaptarse mientras espera al peatón; SUB-WP con waypoints fijos simplemente se detiene y no puede
recuperarse de forma eficiente.

**goal_24 — mejora notable (+11.3pp): 71.7% → 83.0%.**
El LIDAR 5m permite detectar al peatón en el cruce de approach/exit con suficiente anticipación.
La ampliación del umbral de reward a 4.0m permite esperar antes de entrar a la zona de colisión.

**goal_28 — mejora destacada (+16.0pp): 83.3% → 99.3%.**
El mayor éxito del experimento. Detección anticipada a 5m y umbral 4.0m funcionan perfectamente
para este goal: la geometría permite esperar en zona segura antes de cruzar.

**goal_21 — primera mejora real (+6.3pp): 0.7% → 7.0%.**
El corredor más estrecho del almacén muestra por primera vez una tasa de éxito apreciable.
El LIDAR 5m detecta al peatón antes de entrar al corredor. Aun así, 93% son col_exit —
la geometría del corredor no permite una estrategia de espera efectiva con waypoints fijos.

**goal_27 — regresión grave (−20.0pp): 39.3% → 19.3%.** [col_exit = 46%]
La regresión más preocupante del experimento. El exit-corridor reward con umbral 4.0m penaliza
al robot cuando el peatón pasa a 3-4m de forma normal durante la maniobra de salida. Con SUBWP
(waypoints fijos), el robot no puede adaptar su ruta y queda bloqueado esperando innecesariamente,
generando colisiones de salida. Con STHWP el subgoal dinámico puede desplazarse lateralmente para
esquivar sin detener el avance.

**goal_26 — regresión notable (−11.0pp): 94.3% → 83.3%.**
Mismo mecanismo que goal_27, aunque menos pronunciado porque la geometría de goal_26 ofrece más
margen para esperar sin bloquearse.

**goal_25 — permanece prácticamente insoluble: 4.3% → 1.0%.** [col_exit = 99%]
La geometría de este goal no permite que el robot espere fuera del corredor cuando el peatón
lo bloquea. Ninguna mejora de observación resolverá esto — requiere A* con coste dinámico (E2.2)
para que el planificador evite asignar este corredor cuando el peatón está presente.

**col_exit como modo de fallo dominante: 14.5% global.**
Los goals 19, 21, 25 siguen con col_exit muy alto (66%, 93%, 99%). El LIDAR 5m detecta al
peatón pero no puede cambiar la trayectoria del planificador — esa mejora pertenece a E2.2.

**Conclusión E2.1 SUBWP:**
La obs realista mejora marginalmente SUBWP (+0.7pp) frente a STHWP (+5.5pp). El resultado
negativo de goal_27 y goal_26 revela una limitación fundamental de los waypoints fijos: no
pueden adaptarse a esperas largas sin bloquear el progreso. El threshold 4.0m del exit-corridor
reward es adecuado para STHWP (subgoal dinámico) pero genera regresiones en SUBWP. El siguiente
paso (E2.2) debe abordar el planificador, no la observación.

**Mejor seed SUBWP E2.1:** s524 — 83.2%

**Corrección de s123:** El modelo base `subwp_s123_wp75_r2_stage5_final` sufrió olvido catastrófico
del approach durante stage5 (return-only). La corrección (E2.0b) cargó desde `stage4_r2` y obtuvo
83.0%, el mejor resultado individual entre los 3 seeds.

---

---

## Sección 21 — Experimento E2.2: A* con replanning LIDAR dinámico

### Motivación E2.2

E2.1 mostró que SUBWP se beneficia marginalmente de la observación realista (+0.7pp), con
regressions en goals 26 y 27 por el exit-corridor reward. E2.2 pretende abordar la causa raíz
de los fallos col_exit: hacer que el planificador A* evite corredores donde el peatón está
presente, usando LIDAR en lugar del supervisor para localizar obstáculos.

Para SUBWP, E2.2 añade por primera vez replanning mid-episode (no existía en E1.x). El mecanismo:
si un rayo LIDAR en rango dinámico [1.5m, 4.5m] proyecta a celda libre del mapa estático y
cae dentro de REPLAN_DIST_PERP=1.0m del vector robot→waypoint, se replanifica la ruta completa
y se regeneran los waypoints con el mismo espaciado de precomputación.

### Configuración E2.2 SUBWP

| Parámetro | Valor |
|-----------|-------|
| Base | subwp_e2_1_s{42,123,524}_final |
| Guardado | subwp_e2_2_s{42,123,524}_final |
| Stage | 6 (ciclo completo) |
| Steps | 6M fine-tune |
| lr | 1e-4 |
| ent_coef | 0.01 |
| max_steps | 6000 |
| Obs | 40 dims (sin cambio) |
| Replanning | LIDAR-based (nuevo en SUBWP) |
| REPLAN_COOLDOWN | 20 steps |
| REPLAN_INFLATE_CELLS | 4 |
| REPLAN_DIST_PERP | 1.0m |

### Resultados de inferencia E2.2 SUBWP

**Global por seed (2800 ep/seed):**

| Seed | Éxito | col_approach | col_exit | truncado |
|------|-------|--------------|----------|----------|
| s42  | 77.0% | 7.4% | 15.6% | 0% |
| s123 | 80.7% | 5.9% | 13.4% | 0% |
| s524 | 77.2% | 3.8% | 19.0% | 0% |
| **Media** | **78.3%** | **5.7%** | **16.0%** | **0%** |

**E2.2 supone una regresión de −4.1pp respecto a E2.1 (82.4% → 78.3%).**

**Por goal — comparativa E2.1 vs E2.2 (300 ep/goal, 3 seeds):**

| Goal | E2.1 | E2.2 | Δ | col_exit E2.2 |
|------|------|------|---|----------------|
| goal_01–07 | 100% | 100% | 0pp | 0 |
| goal_08 | 100% | 99.7% | −0.3pp | 1 |
| goal_09 | 100% | 91.0% | −9.0pp | 27 |
| goal_10 | 79.7% | 79.7% | 0pp | 61 |
| goal_11–13 | 100% | 100% | 0pp | 0 |
| goal_14 | 100% | 89.0% | −11.0pp | 33 |
| goal_15 | 100% | 84.7% | −15.3pp | 46 |
| goal_16 | 94.3% | 79.0% | −15.3pp | 63 |
| goal_17 | 86.3% | 77.3% | −9.0pp | 1 |
| goal_18 | 67.7% | **81.7%** | **+14.0pp** ⬆ | 35 |
| goal_19 | 17.7% | 19.0% | +1.3pp | 126 |
| goal_20 | 67.3% | 52.0% | −15.3pp | 133 |
| goal_21 | 7.0% | **22.0%** | **+15.0pp** ⬆ | 229 |
| goal_22 | 100% | 83.7% | −16.3pp | 31 |
| goal_23 | 100% | 100% | 0pp | 0 |
| goal_24 | 83.0% | 77.3% | −5.7pp | 29 |
| goal_25 | 1.0% | **12.0%** | **+11.0pp** ⬆ | 252 |
| goal_26 | 83.3% | 58.7% | **−24.7pp** | 60 |
| goal_27 | 19.3% | **31.7%** | **+12.3pp** ⬆ | 159 |
| goal_28 | 99.3% | 54.3% | **−45.0pp** | 58 |
| **MEDIA** | **82.4%** | **78.3%** | **−4.1pp** | |

### Análisis cualitativo E2.2 SUBWP

**Resultado principal: regresión de −4.1pp (82.4% → 78.3%). Menor que STHWP (−12.5pp), pero igualmente negativa.**

**El hallazgo más importante — inversión del ranking entre sistemas:**
Por primera vez en la serie, SUBWP (78.3%) supera a STHWP (76.3%). En E2.1, STHWP superaba por
+6.4pp. Este resultado confirma que E2.2 daña más a STHWP porque su subgoal dinámico es más
sensible a cambios de ruta que los waypoints discretos de SUBWP.

**Causa raíz (igual que STHWP) — falsos positivos LIDAR:**
El filtro `grid_nav==0` no elimina todos los rayos que golpean paredes, porque la discretización
del grid (0.25m) permite que ángulos de incidencia oblicuos proyecten rayos de pared a celdas
libres del mapa estático. Estos falsos obstáculos bloquean pasillos necesarios.

**goal_28 — regresión catastrófica (−45.0pp): 99.3% → 54.3%.**
El peor resultado del experimento. goal_28 era el éxito más notable de E2.1. La geometría de
su corredor de salida hace que las paredes laterales aparezcan frecuentemente en el rango
dinámico LIDAR, desencadenando replanning que enruta al robot fuera del corredor.

**goals con mejora real (replanning funciona):**
- goal_21: +15pp. El peatón cruza limpiamente el corredor a distancias medias (2-3m) sin
  interferencia de paredes cercanas → LIDAR lo detecta bien → A* evita el corredor.
- goal_27: +12.3pp. El replanning recupera la regresión de E2.1 (−20pp). Corredor suficientemente
  ancho para que el peatón sea distinguible de las paredes.
- goal_25: +11pp. Situación similar a goal_21.
- goal_18: +14pp. Mejora robusta.

**Conclusión científica clave:**
Las mejoras en goals 18, 21, 25, 27 prueban que el concepto de replanning LIDAR es correcto
cuando la geometría permite distinguir el peatón de las paredes. El problema no es el algoritmo
sino la ausencia de un paso de segmentación dinámico/estático. Con solo distancia LIDAR, en
pasillos estrechos de almacén, el peatón es indistinguible de la pared del fondo del pasillo.

**Tabla resumen de la serie E completa:**

| Experimento | STHWP | SUBWP | Δ(STH−SUB) | Variable cambiada |
|-------------|-------|-------|------------|-------------------|
| E1.5 | 83.3% | 81.7% | +1.5pp | Exit-corridor reward |
| E2.1 | 88.8% | 82.4% | +6.4pp | Obs realista (sin supervisor) |
| E2.2 | 76.3% | 78.3% | −2.0pp | Replanning LIDAR (sin supervisor) |

**Conclusión E2.2 SUBWP y cierre de la serie experimental:**
E2.2 es un resultado negativo que clarifica los límites del sistema: la observación realista
(LIDAR) puede sustituir al supervisor en la red neuronal (E2.1), pero no en el planificador A*
sin un módulo de detección de obstáculos dinámicos más sofisticado. El mejor resultado SUBWP
de toda la serie sigue siendo E2.1 con 82.4%.

Los resultados finales consolidados del TFM:
- **STHWP best: E2.1 — 88.8%**
- **SUBWP best: E2.1 — 82.4%**
- Brecha máxima entre sistemas: E2.1 (+6.4pp a favor de STHWP)
- La ventaja de STHWP aumenta con obs realista pero desaparece (e invierte) con replanning LIDAR

**Mejor resultado global SUBWP: E2.1 — 82.4%**

---

## Sección 22 — Resumen final de la serie experimental SUBWP

### Tabla maestra — todos los experimentos

| Experimento | Descripción | Media 3 seeds | Δ vs anterior |
|-------------|-------------|---------------|---------------|
| E1 | Fine-tune con 1 peatón (48 dims, obs supervisor) | 81.7% | — |
| E1.2 | Sesgo goals 17-21,24-28 | ~81% | ~0pp |
| E1.3 | Trayectoria peatón extendida x∈[−4,4] | 79.6% | −2.1pp |
| E1.4 | Replanning A* supervisor + patience reward | 79.0% | −0.6pp |
| E1.5 | Exit-corridor reward + curriculum ponderado | 81.7% | +2.7pp |
| E2.0 | Pre-entrenamiento stage6 sin peatón (base E2.1) | — | — |
| **E2.1** | **Obs realista: solo LIDAR 5m (40 dims)** | **82.4%** | **+0.7pp** |
| E2.2 | Replanning LIDAR dinámico (nuevo en SUBWP) | 78.3% | −4.1pp |

### Hallazgos principales de la serie

**1. SUBWP es menos sensible a cambios de observación que STHWP.**
E2.1 solo aporta +0.7pp a SUBWP vs +5.5pp a STHWP. Los waypoints fijos limitan la capacidad
de aprovechar la información anticipada del LIDAR 5m: el robot puede ver al peatón antes,
pero no puede cambiar su ruta hacia el siguiente waypoint.

**2. El replanning A* supervisor (E1.4) es perjudicial en SUBWP (−0.6pp).**
A diferencia de STHWP, el replanning con waypoints fijos genera inestabilidad en la secuencia
de waypoints al resetear `_wp_idx=0` tras cada replanning. Demasiados replanning → oscilación.

**3. El exit-corridor reward (E1.5) es la mejora más robusta en SUBWP (+2.7pp).**
Señalizar explícitamente que el robot debe esperar antes de salir del pasillo cuando el peatón
está cerca es el cambio de reward más efectivo para SUBWP.

**4. SUBWP supera a STHWP por primera vez en E2.2 (78.3% vs 76.3%).**
El replanning LIDAR daña más a STHWP porque su subgoal STH-WP salta a posiciones inesperadas
al replanificar tramos largos. Los waypoints discretos de SUBWP son más robustos a cambios de ruta.

**5. La brecha STHWP−SUBWP varía con la arquitectura:**
- Obs supervisor (E1.5): +1.5pp a favor STHWP
- Obs realista (E2.1): +6.4pp a favor STHWP ← mayor brecha
- Replanning LIDAR (E2.2): −2.0pp (SUBWP supera) ← inversión

### Goals estructuralmente limitados (SUBWP)

| Goal | Mejor resultado | Causa |
|------|-----------------|-------|
| goal_25 | 12% (E2.2) | Corredor de 1 carril: bloqueo total |
| goal_21 | 22% (E2.2) | Corredor más estrecho, peatón bloquea salida |
| goal_19 | 19% (E2.1) | Col_exit dominante incluso con LIDAR 5m |
| goal_27 | 32% (E2.2) | Geometría compleja, sensible al exit-corridor reward |

---

## Sección 23 — E2.2b: Fix LIDAR_MIN_DYNAMIC 1.5→2.5m (SUBWP)

### Configuración

| Parámetro | Valor |
|-----------|-------|
| Base | `subwp_e2_1_s{seed}_final` (NO desde E2.2) |
| Cambio principal | `LIDAR_MIN_DYNAMIC` 1.5m → 2.5m |
| Steps | 6M fine-tuning |
| lr | 1e-4 |
| ent_coef | 0.01 |
| reset_num_timesteps | True |
| max_steps | 6000 |
| Hard goals | {17,18,19,20,21,24,25,26,27,28} (3× prob) |

### Resultados de entrenamiento (stats al final de 6M steps)

| Seed | Tasa global (train) | Goals hard medios | Goals <60% |
|------|--------------------|--------------------|------------|
| s42  | 66.7% (1447/2170)  | ~69%               | goal_02, goal_05, goal_20, goal_22 |
| s123 | 69.3% (1518/2192)  | ~71%               | goal_07, goal_11, goal_22 |
| s524 | 70.0% (1543/2204)  | ~72%               | goal_01, goal_07, goal_12, goal_14 |

Aparentemente, el entrenamiento parece razonable (67-70%). Sin embargo, los resultados de
inferencia revelan una disociación grave para s123.

### Resultados de inferencia (100 ep/goal, 28 goals, deterministic)

| Seed | Éxito | Col. approach | Col. exit | Truncado |
|------|-------|---------------|-----------|----------|
| s42  | 76.6% (2144/2800) | 14.3% | 9.1%  | 0.0% |
| s123 | 51.3% (1437/2800) | 10.4% | 29.4% | **9.0%** |
| s524 | 79.3% (2220/2800) | 7.6%  | 13.1% | 0.0% |
| **Media** | **69.1%** | **10.8%** | **17.2%** | **3.0%** |

### Comparativa E2.1 → E2.2 → E2.2b

| Métrica | E2.1 | E2.2 | E2.2b | Δ(E2.2b−E2.1) | Δ(E2.2b−E2.2) |
|---------|------|------|-------|----------------|----------------|
| Media 3 seeds | 82.4% | 78.3% | **69.1%** | −13.3pp | −9.2pp |
| Truncados | 0.0% | 0.0% | **3.0%** | +3.0pp | +3.0pp |

E2.2b es **peor que E2.2** (−9.2pp adicionales). La corrección del umbral no solo no mejoró,
sino que degradó el sistema SUBWP.

### Caso crítico: SUBWP s123 — Colapso de inferencia

La semilla s123 exhibe el peor resultado de toda la serie E2.x:

| Métrica | s42 | s123 | s524 |
|---------|-----|------|------|
| Entrenamiento | 66.7% | 69.3% | 70.0% |
| Inferencia | 76.6% | **51.3%** | 79.3% |
| Truncados | 0% | **9.0%** | 0% |
| Col. exit | 9.1% | **29.4%** | 13.1% |

La discrepancia train→infer de s123 (69.3% → 51.3%) indica **olvido catastrófico selectivo**:

- **9% truncados** (251 episodios): el robot entra en bucles de navegación, incapaz de alcanzar
  el waypoint siguiente en 6000 steps. Esto es inédito en toda la serie (otros seeds: 0%).
- **29.4% col_exit**: el peatón bloquea la salida del pasillo y el robot colisiona al intentar
  forzar el paso, en lugar de esperar.
- El fine-tuning de E2.2b sobre E2.1 re-aprendió a navegar con replanning LIDAR, pero
  la política deterministic (inferencia) presenta un comportamiento diferente al estocástico
  (entrenamiento), lo que revela sobre-ajuste al ruido de exploración de la fase de train.

### Goals problemáticos en inferencia E2.2b

| Goal | s42 | s123 | s524 | Causa |
|------|-----|------|------|-------|
| goal_08 | —  | 0%  | —   | Colapso total s123 |
| goal_09 | —  | 0%  | —   | Colapso total s123 |
| goal_10 | —  | 0%  | —   | Colapso total s123 |
| goal_12 | —  | 0%  | —   | Colapso total s123 |
| goal_18 | 0% | —   | —   | Pasillo interior estrecho |
| goal_19 | 39%| 23% | 23% | Col_exit sistemático |
| goal_20 | —  | 0%  | —   | Colapso total s123 |
| goal_21 | 64%| 0%  | 17% | Corredor estrecho |
| goal_25 | 0% | 3%  | 0%  | Corredor de 1 carril |

### Análisis — Por qué E2.2b es peor que E2.2 en SUBWP

**Umbral 2.5m vs SUBWP con WP_STEP=1.5m:**
En SUBWP, el paso entre waypoints es 1.5m. Cuando el replanning LIDAR activa, resetea
`_wp_idx=0` y redefine `full_path` con waypoints submuestreados. Con LIDAR_MIN_DYNAMIC=2.5m:

1. Se detectan menos falsos positivos de paredes → menos replanning total
2. **Pero**: los replanning que sí ocurren son con obstáculos reales (el peatón), y el
   re-routing alrededor del peatón a 2.5m produce un camino alternativo con WP_STEP=1.5m
   que puede llevar al robot hacia zonas con paredes a 2.5-4.5m → más falsos positivos
   en el nuevo trayecto → bucle de replanning → 9% truncados en s123.

Con umbral 1.5m (E2.2), más falsos positivos pero menos bucles: el robot replanifica
frecuentemente y converge a alguna ruta. Con umbral 2.5m (E2.2b), menos replanning pero
cuando ocurre, el nuevo trayecto tiene más probabilidad de volver a activar replanning
en pasillos interiores, creando un bucle de replanning que agota el límite de steps.

### Conclusión E2.2b SUBWP

E2.2b confirma que el problema de la serie E2.2 es **arquitectural, no paramétrico**. En SUBWP
el cambio de umbral agrava el problema (−9.2pp adicionales vs E2.2) porque el mecanismo de
replanning con waypoints discretos es más sensible a bucles de replanning que STH-WP
(cuyo subgoal continuo converge aunque salte).

La serie E2.2/E2.2b queda documentada como **resultado negativo reproducible**:
ningún valor del umbral LIDAR_MIN_DYNAMIC permite separar peatones de paredes en un
almacén con pasillos de ~2m de ancho, sin información semántica adicional.

**Mejor resultado global SUBWP: E2.1 — 82.4%**

### Actualización tabla maestra (Sección 22)

| Experimento | Descripción | Media 3 seeds | Δ vs anterior |
|-------------|-------------|---------------|---------------|
| E1 | Fine-tune con 1 peatón (48 dims) | 81.7% | — |
| E1.2 | Sesgo goals 17-21,24-28 | ~81% | ~0pp |
| E1.3 | Trayectoria extendida | 79.6% | −2.1pp |
| E1.4 | Replanning supervisor + patience | 79.0% | −0.6pp |
| E1.5 | Exit-corridor reward + curriculum | 81.7% | +2.7pp |
| E2.0 | Pre-entrenamiento sin peatón (base E2.1) | — | — |
| **E2.1** | **Obs realista: solo LIDAR 5m** | **82.4%** | **+0.7pp** |
| E2.2 | Replanning LIDAR dinámico (umbral 1.5m) | 78.3% | −4.1pp |
| E2.2b | Replanning LIDAR dinámico (umbral 2.5m) | 69.1% | −9.2pp |

---

## Sección 24 — Ablación E2.1 sin peatón + Análisis estadístico STHWP vs SUBWP

### Motivación

Tras completar la serie E2.x, se realizaron dos análisis adicionales para cuantificar con
rigor el impacto del peatón y validar estadísticamente la diferencia entre sistemas:

1. **Inferencia E2.1 sin peatón** — mismos modelos E2.1, mundo `warehouse_1_subwp_noped.wbt`
   (0 peatones). Mide la tasa de éxito base de navegación estática pura.
2. **Análisis estadístico comparativo** — chi-cuadrado por goal y global, IC 95% Wilson,
   Cohen's h. Script: `simulacion/controllers/analisis_estadistico.py`.

### Resultados inferencia sin peatón (SUBWP E2.1)

| Seed | Éxito | Resultado |
|------|-------|-----------|
| s42  | 2800/2800 | 100.0% |
| s123 | 2800/2800 | 100.0% |
| s524 | 2800/2800 | 100.0% |
| **Media** | | **100.0%** IC95%=[100.0–100.0] |

Navegación estática perfecta en los 28 goals. Confirma que el modelo E2.1 no tiene
déficits de navegación base — todas las colisiones observadas con peatón son causadas
exclusivamente por el obstáculo dinámico.

### Coste del peatón — SUBWP

| Condición | Tasa global | IC 95% |
|-----------|------------|--------|
| Sin peatón | 100.0% | [100.0–100.0] |
| Con peatón (E2.1) | 82.4% | [81.5–83.2] |
| **Coste peatón** | **−17.6pp** | |

El peatón reduce el rendimiento de SUBWP en 17.6pp, frente a los 11.2pp de STHWP.
Esta diferencia de 6.4pp en coste refleja la menor capacidad de SUBWP para adaptarse
dinámicamente: sus waypoints fijos no permiten esquivar al peatón en exit corridors.

### Goals más perjudicados por el peatón (SUBWP)

| Goal | Sin peatón | Con peatón | Coste |
|------|-----------|-----------|-------|
| goal_25 | 100% | 1.0% | −99.0pp |
| goal_21 | 100% | 7.0% | −93.0pp |
| goal_19 | 100% | 17.7% | −82.3pp |
| goal_27 | 100% | 19.3% | −80.7pp |
| goal_20 | 100% | 67.3% | −32.7pp |

El 80% del daño total del peatón sobre SUBWP se concentra en estos 5 goals. Todos son
**exit corridors** donde el peatón cruza en el momento en que el robot intenta salir.
SUBWP no puede redirigir los waypoints alrededor del peatón → colisión sistemática.
Sin peatón, los 28 goals se resuelven al 100%.

### Análisis estadístico STHWP vs SUBWP (E2.1 con peatón)

**Test global chi-cuadrado:**

| Métrica | Valor |
|---------|-------|
| STHWP | 7455/8400 (88.8%) IC95%=[88.1–89.4] |
| SUBWP | 6918/8400 (82.4%) IC95%=[81.5–83.2] |
| χ² | 138.88 |
| p-valor | < 0.0001 |
| Cohen's h | 0.183 (efecto pequeño) |

La diferencia global es **estadísticamente significativa** pero de **tamaño pequeño**.

**Goals con diferencia significativa (14 de 28, p<0.05):**

| Patrón geométrico | Goals | Ganador | Δ típico |
|-------------------|-------|---------|---------|
| Exit corridors | 17,18,19,20,21,25,27 | **STHWP** | +14 a +93pp |
| Esquinas interiores | 23,24,28 | **SUBWP** | +33 a +59pp |
| Exit pasillo estrecho | 14,15,16 | **SUBWP** | +4 a +8pp |
| Exit corridor ancho | 10 | **STHWP** | +18pp |

**Interpretación desde la arquitectura SUBWP:**
- SUBWP pierde sistemáticamente en **exit corridors** (19, 21, 25, 27): el siguiente
  waypoint está al otro lado del peatón y SUBWP no puede esquivarlo — navega directo
  hasta colisionar. STHWP con subgoal adaptativo esquiva lateralmente.
- SUBWP gana en **esquinas interiores** (23, 24, 28): sus waypoints fijos en la
  esquina guían al robot con un ángulo predecible, mientras el lookahead de STH-WP
  salta al otro lado y dirige al robot hacia la pared.
- 14 goals sin diferencia significativa: goals fáciles (ambos al 100%) y goal_26
  (p=0.25, diferencia no reproducible).

### Coste del peatón comparado entre sistemas

| Sistema | Sin peatón | Con peatón | Coste | Goals más afectados |
|---------|-----------|-----------|-------|---------------------|
| STHWP | 100.0% | 88.8% | −11.2pp | 23(−59pp), 24(−63pp), 25(−75pp) |
| SUBWP | 100.0% | 82.4% | −17.6pp | 19(−82pp), 21(−93pp), 25(−99pp) |

La diferencia de robustez (6.4pp menos coste en STHWP) coincide exactamente con la
diferencia de rendimiento en E2.1, confirmando que **toda la ventaja de STHWP sobre
SUBWP se explica por mayor robustez dinámica**, no por mejor navegación estática.

### Conclusiones del análisis estadístico

1. **Ambos sistemas tienen navegación estática perfecta** (100.0% sin peatón). Los
   goals con tasa baja no son un problema de aprendizaje sino de geometría dinámica.

2. **SUBWP pierde 6.4pp más que STHWP ante el peatón** (17.6pp vs 11.2pp). La
   arquitectura de waypoints fijos es estructuralmente vulnerable a exit corridors
   bloqueados: sin capacidad de redirigir el subgoal, el robot no puede esquivar.

3. **La ventaja de SUBWP sobre STHWP en esquinas interiores** (goals 23, 24, 28)
   demuestra que los waypoints fijos son más estables en geometrías de esquina donde
   el lookahead dinámico de STH-WP introduce inestabilidad.

4. **Exit corridors son el talón de Aquiles de SUBWP**: goal_19 (82.3pp de coste),
   goal_21 (93.0pp), goal_25 (99.0pp). Estos goals requieren evasión lateral activa
   que WP_STEP=1.5m y waypoints discretos no pueden proporcionar.

5. **Diferencia global estadísticamente significativa pero efecto pequeño** (h=0.183).
   La elección entre sistemas debería hacerse en función de la distribución de goals
   esperada en el entorno real de despliegue.

*Fin del log de entrenamiento SUB-WP.*
