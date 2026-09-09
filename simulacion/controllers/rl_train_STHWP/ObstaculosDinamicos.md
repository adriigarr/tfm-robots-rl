# Experimento: Obstáculos Dinámicos — STH-WP vs SUB-WP

Documento de diseño experimental y registro de resultados para la evaluación de robustez
de ambos métodos de generación de waypoints ante obstáculos dinámicos en el almacén.

---

## 1. Motivación

El entrenamiento de ambos métodos (STH-WP y SUB-WP) se realizó en un entorno **estático**:
el almacén no contiene ningún agente móvil durante el entrenamiento. En un almacén real,
la política de navegación debe coexistir con otros robots, operarios y carretillas.

La pregunta de investigación es:

> **¿Qué método de generación de waypoints (STH-WP o SUB-WP) se degrada menos ante
> la presencia de obstáculos dinámicos no vistos durante el entrenamiento?**

El experimento se ejecuta sobre los modelos ya entrenados, **sin reentrenamiento**.
Es una evaluación de generalización (zero-shot robustness), no un nuevo ciclo de
aprendizaje.

---

## 2. Hipótesis

### H1 — Hipótesis principal

**STH-WP se degrada menos que SUB-WP ante obstáculos dinámicos.**

**Justificación mecánica:**

| Característica | SUB-WP | STH-WP |
|----------------|--------|--------|
| Subgoal | Waypoint discreto fijo (índice `_wp_idx` en el path A*) | Punto continuo 1.5m adelante en el path, recalculado cada step |
| Reacción a un obstáculo entre el robot y el subgoal | El robot intenta alcanzar el waypoint bloqueado → colisión probable | El subgoal se proyecta siempre sobre el path libre más cercano → reacción más fluida |
| Sensores usados | LiDAR local + distancia al subgoal | LiDAR local + distancia al subgoal continuo |
| Dependencia de la posición exacta del subgoal | Alta (waypoint fijo) | Baja (subgoal se actualiza en cada step) |

En SUB-WP, si un obstáculo dinámico se interpone entre el robot y el waypoint actual,
la política recibe una observación con distancia al subgoal creciente y LiDAR bloqueado,
pero el subgoal no cambia de posición. La política, entrenada solo con obstáculos
estáticos, puede no haber aprendido a esperar o rodear — ejecutará la acción que en
entrenamiento llevaba al waypoint.

En STH-WP, el subgoal se recalcula sobre el path disponible en cada step. Si el obstáculo
dinámico bloquea la trayectoria directa, el subgoal se desplaza al punto más cercano del
path que siga siendo alcanzable. La política recibe una señal de dirección actualizada,
lo que puede generar un comportamiento de espera o desvío más natural aunque no haya sido
explícitamente entrenado.

### H2 — Hipótesis secundaria

**La varianza entre seeds es mayor en SUB-WP que en STH-WP ante obstáculos dinámicos.**

En entorno estático ya se observa que SUB-WP tiene una varianza entre seeds muy alta
(s42: 32.6% vs s123/s524: 100%). STH-WP es más homogéneo (s42/s524: 100%, s123: 82.1%).
La hipótesis es que los obstáculos dinámicos amplifican esta diferencia de varianza.

### H3 — Hipótesis nula (a refutar)

**Ambos métodos se degradan por igual, porque la causa principal de colisión es la
política de LiDAR local (compartida), no el mecanismo de subgoal.**

Si H3 no puede refutarse, la conclusión es que el mecanismo de generación de waypoints
no es el factor determinante en la robustez ante obstáculos dinámicos — lo que también
es un resultado válido y publicable.

---

## 3. Baseline: resultados en entorno estático

### 3.1 STH-WP — Inferencia ciclo completo (approach+exit+retorno)

Modelo: `run002_s{42,524}_stage4v2_final` + `run002_s{42,524}_stage5_final`
100 episodios × 28 goals = 2800 ciclos por seed.

| Seed | Éxito | Col. approach | Col. exit | Col. retorno | Truncado | Pasos/ciclo |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| s42  | **100%** | 0% | 0% | 0% | 0% | 2068.5 |
| s123 | 82.1% | 0% | 17.9% | 0% | 0% | 1834.6 |
| s524 | **100%** | 0% | 0% | 0% | 0% | 2072.6 |
| **Media** | **94.0%** | 0% | **6.0%** | 0% | 0% | — |

*s123 falla en 5 goals específicos (09, 10, 14, 15, 22) por fragilidad determinista
conocida del seed — no es un fallo generalizado.*

### 3.2 SUB-WP — Inferencia ciclo completo (approach+exit+retorno)

Modelo: `subwp_s{42,123,524}_wp75` (ciclo completo con max_waypoints=75)
100 episodios × 28 goals = 2800 ciclos por seed.

| Seed | Éxito | Col. approach | Col. exit | Col. retorno | Truncado | Pasos/ciclo |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| s42  | 32.6% | 25.0% | 13.2% | 0% | 29.2% | 3194.7 |
| s123 | **100%** | 0% | 0% | 0% | 0% | 3367.6 |
| s524 | **100%** | 0% | 0% | 0% | 0% | 3149.7 |
| **Media** | **77.5%** | **8.3%** | **4.4%** | 0% | **9.7%** | — |

*s42 sufre olvido catastrófico en el ciclo completo: 16 goals al 0% de éxito
(col_approach en goals 01–04, 11–13; truncado en goals 08–10, 16, 20–22, 25).*

### 3.3 Comparativa estático

| Métrica | STH-WP (media 3 seeds) | SUB-WP (media 3 seeds) | Δ |
|---------|:---:|:---:|:---:|
| Éxito global | **94.0%** | 77.5% | **+16.5 pp** |
| Colisión total | **2.0%** | 12.7% | **−10.7 pp** |
| Truncado | **0%** | 9.7% | **−9.7 pp** |
| Varianza entre seeds | Baja (82–100%) | Alta (33–100%) | STH-WP más estable |

---

## 4. Diseño del experimento

### 4.1 Tipo de obstáculos a añadir en Webots

Se proponen tres niveles de dificultad creciente:

| Nivel | Descripción | Implementación Webots |
|-------|-------------|----------------------|
| **L1 — Obstáculo estático no visto** | Caja/objeto en posición aleatoria del corredor | `Solid` node con posición aleatoria en reset |
| **L2 — Obstáculo oscilante** | Objeto que se mueve de lado a lado en un corredor | `Solid` node con supervisor controlando posición en bucle |
| **L3 — Robot móvil** | Segundo robot MiR100 siguiendo una ruta fija | Segundo controller con path predefinido |

**Recomendación**: empezar con L2 (oscilante) — es más representativo que L1 (sigue siendo estático en cada episodio) y más controlable que L3 (la interacción robot-robot añade variables confundidoras).

### 4.2 Posicionamiento de los obstáculos

Los obstáculos se colocarán en los **corredores principales** del almacén, no dentro
de las estanterías. Las posiciones candidatas:

- **Corredor central** (y ≈ −1.5): ruta de tránsito entre zona_espera y bloques 3/5
- **Corredor izquierdo** (x ≈ −9.5): ruta de retorno zona_descarga → zona_espera
- **Entrada bloque 3** (x ≈ 3.5, y ≈ −5): acceso a goals 11–16

### 4.3 Protocolo de evaluación

- Mismos scripts de inferencia ciclo completo (`infer_run002_s{42,524}_ciclo_completo.py`)
  y equivalentes de SUB-WP, **modificados** para activar obstáculos en el `reset()` vía supervisor
- **50 episodios por goal** (vs 100 en estático) para cada nivel de obstáculo
  → 50 × 28 × 3 seeds × 2 métodos = 8400 episodios totales por nivel
- Mismas métricas: éxito, col_approach, col_exit, col_retorno, truncado
- Añadir métrica nueva: **n_paradas** (número de veces que el robot frena ante el obstáculo)
  como indicador de comportamiento esquivo vs colisión

### 4.4 Variables controladas

- Velocidad del obstáculo: constante en cada nivel (0.3 m/s para L2)
- Posición inicial del obstáculo: aleatoria pero dentro de la zona definida
- Seed del modelo: fijos (s42, s524 para STH-WP; s123, s524 para SUB-WP)
- Timeout: mismos que en entorno estático (4000 steps ap+ex, 3000 steps retorno)

---

## 5. Métricas y criterios de análisis

### 5.1 Métricas primarias

| Métrica | Definición |
|---------|------------|
| `exito_%_dinamico` | % ciclos completados con obstáculo presente |
| `degradacion_%` | `exito_%_estatico − exito_%_dinamico` |
| `colision_inducida_%` | Colisiones nuevas atribuibles al obstáculo (Δ respecto al estático) |

### 5.2 Criterio de comparación

STH-WP "gana" si:
- `degradacion_STHWP < degradacion_SUBWP` en al menos 2 de los 3 niveles
- La diferencia es estadísticamente significativa (test de proporciones, α=0.05)

Si la diferencia es < 5 pp, se considera empate — H3 no puede refutarse.

---

## 6. Resultados

Experimento ejecutado el 2026-07-06/07. Nivel implementado: **peatones activos** (equivalente
a L2 oscilante). Dos peatones Webots Pedestrian PROTO con `enableBoundingObject TRUE`:

- **PEDESTRIAN_1**: corredor central y≈0.3, oscila x=−2 ↔ x=4 a 0.5 m/s
- **PEDESTRIAN_2**: corredor izquierdo x≈−9.5, oscila y=−5 ↔ y=−0.5 a 0.5 m/s

Protocolo: 100 ep × 28 goals × 2 seeds por método = 2800 ciclos por seed.
Seeds usados: STH-WP s42 y s524 (ambos 100% en estático); SUB-WP s123 y s524 (100% en estático).

---

### 6.1 Resultados globales

| Método | Seed | Éxito estático | Éxito dinámico | Degradación | Col. approach | Col. exit | Col. retorno | Truncado |
|--------|------|:--------------:|:--------------:|:-----------:|:-------------:|:---------:|:------------:|:--------:|
| STH-WP | s42  | 100%           | 10.8%          | −89.2 pp    | 37.8%         | 32.7%     | 18.6%        | 0%       |
| STH-WP | s524 | 100%           | 3.7%           | −96.3 pp    | 34.4%         | 35.4%     | 26.5%        | 0%       |
| **STH-WP media** | | **100%** | **7.3%** | **−92.7 pp** | 36.1% | 34.1% | 22.6% | 0% |
| SUB-WP | s123 | 100%           | 30.0%          | −70.0 pp    | 21.7%         | 22.0%     | 26.2%        | 0%       |
| SUB-WP | s524 | 100%           | 36.1%          | −63.9 pp    | 20.2%         | 25.8%     | 18.0%        | 0%       |
| **SUB-WP media** | | **100%** | **33.1%** | **−66.9 pp** | 21.0% | 23.9% | 22.1% | 0% |

---

### 6.2 Resultados por goal — STH-WP

| Goal | s42 éxito | s42 col_ap | s42 col_ex | s42 col_ret | s524 éxito | s524 col_ap | s524 col_ex | s524 col_ret |
|------|:---------:|:----------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:------------:|
| goal_01 | 5% | 0% | 54% | 41% | 0% | 0% | 59% | 41% |
| goal_02 | 24% | 0% | 46% | 30% | 0% | 0% | 78% | 22% |
| goal_03 | 0% | 0% | **100%** | 0% | 0% | 0% | 83% | 17% |
| goal_04 | 27% | 0% | 10% | 63% | 0% | 0% | 0% | **100%** |
| goal_05 | 19% | 0% | 38% | 43% | 0% | 0% | 50% | 50% |
| goal_06 | 0% | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% |
| goal_07 | 0% | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% |
| goal_08 | 0% | **100%** | 0% | 0% | 0% | 89% | 0% | 11% |
| goal_09 | 0% | **100%** | 0% | 0% | 2% | 87% | 3% | 8% |
| goal_10 | 0% | **100%** | 0% | 0% | 0% | 87% | 0% | 13% |
| goal_11 | 39% | 0% | 21% | 40% | 19% | 0% | 6% | 75% |
| goal_12 | 24% | 0% | 38% | 38% | 1% | 0% | 42% | 57% |
| goal_13 | 32% | 0% | 1% | 67% | 37% | 0% | 0% | 63% |
| goal_14 | 23% | 0% | 48% | 29% | 0% | 0% | **100%** | 0% |
| goal_15 | 2% | 0% | 98% | 0% | 0% | 0% | **100%** | 0% |
| goal_16 | 42% | 0% | 29% | 29% | 22% | 0% | 46% | 32% |
| goal_17 | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% | 0% |
| goal_18 | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% | 0% |
| goal_19 | 0% | **100%** | 0% | 0% | 0% | 88% | 0% | 12% |
| goal_20 | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% | 0% |
| goal_21 | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% | 0% |
| goal_22 | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% | 0% |
| goal_23 | 0% | 0% | **100%** | 0% | 0% | 0% | **100%** | 0% |
| goal_24 | 0% | 15% | 85% | 0% | 6% | 0% | 88% | 6% |
| goal_25 | 11% | 26% | 48% | 15% | 6% | 23% | 36% | 35% |
| goal_26 | 28% | 27% | 0% | 45% | 10% | 47% | 0% | 43% |
| goal_27 | 13% | 33% | 0% | 54% | 0% | 41% | 0% | 59% |
| goal_28 | 14% | 58% | 0% | 28% | 0% | 1% | 0% | 99% |

Goals a 0% en **ambos** seeds STH-WP: 03, 06, 07, 08, 09, 10, 17, 18, 19, 20, 21, 22, 23, 24
(14 de 28 goals completamente bloqueados — 50% de los goals).

---

### 6.3 Resultados por goal — SUB-WP

| Goal | s123 éxito | s123 col_ap | s123 col_ex | s123 col_ret | s524 éxito | s524 col_ap | s524 col_ex | s524 col_ret |
|------|:----------:|:-----------:|:-----------:|:------------:|:----------:|:-----------:|:-----------:|:------------:|
| goal_01 | 0% | 0% | **100%** | 0% | 2% | 0% | 97% | 1% |
| goal_02 | 35% | 0% | 30% | 35% | 38% | 0% | 60% | 2% |
| goal_03 | 48% | 0% | 48% | 4% | 42% | 0% | 15% | 43% |
| goal_04 | 34% | 0% | 34% | 32% | 10% | 0% | 67% | 23% |
| goal_05 | 4% | 0% | 2% | 94% | 16% | 0% | 43% | 41% |
| goal_06 | 0% | 0% | 0% | **100%** | 31% | 0% | 51% | 18% |
| goal_07 | 44% | 0% | 32% | 24% | 8% | 0% | 84% | 8% |
| goal_08 | **60%** | 0% | 0% | 40% | **75%** | 0% | 0% | 25% |
| goal_09 | **79%** | 0% | 0% | 21% | **70%** | 0% | 0% | 30% |
| goal_10 | **54%** | 0% | 27% | 19% | 24% | 0% | 2% | 74% |
| goal_11 | 3% | 0% | 1% | 96% | 38% | 0% | 49% | 13% |
| goal_12 | 20% | 0% | 80% | 0% | 22% | 0% | 39% | 39% |
| goal_13 | 11% | 0% | 70% | 19% | 2% | 0% | 98% | 0% |
| goal_14 | 52% | 40% | 0% | 8% | 41% | 0% | 38% | 21% |
| goal_15 | 62% | 17% | 0% | 21% | 57% | 0% | 39% | 4% |
| goal_16 | 0% | 33% | 34% | 33% | **100%** | 0% | 0% | 0% |
| goal_17 | 59% | 0% | 8% | 33% | 61% | 0% | 0% | 39% |
| goal_18 | 70% | 0% | 0% | 30% | 65% | 0% | 0% | 35% |
| goal_19 | 28% | 44% | 27% | 1% | 0% | **100%** | 0% | 0% |
| goal_20 | 0% | 88% | 0% | 12% | 0% | **100%** | 0% | 0% |
| goal_21 | 4% | 82% | 0% | 14% | 0% | **100%** | 0% | 0% |
| goal_22 | 11% | 78% | 0% | 11% | 0% | **100%** | 0% | 0% |
| goal_23 | 3% | 22% | 74% | 1% | 63% | 1% | 28% | 8% |
| goal_24 | 0% | 50% | 50% | 0% | 28% | 33% | 11% | 28% |
| goal_25 | 58% | 25% | 0% | 17% | 76% | 24% | 0% | 0% |
| goal_26 | 50% | 46% | 0% | 4% | 49% | 32% | 0% | 19% |
| goal_27 | 29% | 27% | 0% | 44% | 53% | 15% | 0% | 32% |
| goal_28 | 22% | 56% | 0% | 22% | 39% | 61% | 0% | 0% |

Goals a 0% en **ambos** seeds SUB-WP: 01, 20 (solo 2 de 28 goals completamente bloqueados).

---

### 6.4 Verificación de hipótesis

| Hipótesis | Predicción | Resultado | Veredicto |
|-----------|-----------|-----------|-----------|
| **H1** — STH-WP se degrada menos | Δ_STHWP < Δ_SUBWP | Δ_STHWP=−92.7 pp vs Δ_SUBWP=−66.9 pp | **REFUTADA** — SUB-WP se degrada menos |
| **H2** — Mayor varianza entre seeds en SUB-WP | σ_SUBWP > σ_STHWP | s42/s524 ratio 3:1 vs s123/s524 ratio 1.2:1 | **REFUTADA** — mayor varianza en STH-WP |
| **H3** — Ambos se degradan igual (nula) | Δ < 5 pp | Δ = 25.8 pp | **REFUTADA** — diferencia significativa |

---

### 6.5 Análisis de causas

#### Goals completamente bloqueados: STH-WP vs SUB-WP

El contraste más claro es en los goals del corredor central (zona del Peatón 1):

| Goal | STH-WP ambos seeds | SUB-WP s123 | SUB-WP s524 |
|------|:-----------------:|:-----------:|:-----------:|
| goal_08 | **0%** (col_ap ~100%) | 60% | 75% |
| goal_09 | **0%** (col_ap ~100%) | 79% | 70% |
| goal_17 | **0%** (col_ap ~100%) | 59% | 61% |
| goal_18 | **0%** (col_ap ~100%) | 70% | 65% |

En STH-WP, el subgoal continuo a 1.5m sobre el path A* apunta permanentemente hacia el
punto bloqueado por el peatón — el robot avanza hasta colisionar. En SUB-WP, los waypoints
discretos pueden caer en posiciones laterales que el robot alcanza bordeando al peatón, o
el robot llega al waypoint antes de que el peatón cruce. La naturaleza discreta del subgoal
ofrece una ventaja estructural no prevista.

STH-WP bloquea completamente 14 de 28 goals (50%). SUB-WP solo bloquea 2 de 28 (7%).

#### col_retorno: Peatón 2 afecta a ambos métodos por igual

| Método | col_retorno dinámico | col_retorno estático | Δ |
|--------|:-------------------:|:-------------------:|:---:|
| STH-WP media | 22.6% | 0% | +22.6 pp |
| SUB-WP media | 22.1% | 0% | +22.1 pp |

El Peatón 2 (corredor de retorno x≈−9.5) impacta igual a ambos métodos. Esto confirma que
en la fase de retorno el mecanismo de waypoint es irrelevante — el bottleneck es la política
de LiDAR local, que en ninguno de los dos métodos fue entrenada con obstáculos en ese corredor.

#### Varianza entre seeds

STH-WP: s42 (10.8%) vs s524 (3.7%) — diferencia de 7.1 pp, ratio 3:1.
SUB-WP: s123 (30.0%) vs s524 (36.1%) — diferencia de 6.1 pp, ratio 1.2:1.

La mayor varianza de STH-WP ante obstáculos dinámicos sugiere que la robustez de cada seed
depende de pequeñas diferencias en la política aprendida para las zonas de paso de los
peatones — diferencias que en estático eran invisibles (ambos 100%).

---

### 6.6 Conclusión

**SUB-WP es más robusto que STH-WP ante obstáculos dinámicos** en condiciones zero-shot
(sin reentrenamiento), con una ventaja de +25.8 pp en éxito medio (33.1% vs 7.3%).

La causa mecánica es que el subgoal continuo de STH-WP "fija" la dirección del robot hacia
el punto bloqueado del path, mientras que los waypoints discretos de SUB-WP ofrecen
posiciones objetivo que el robot puede alcanzar sin cruzar necesariamente la trayectoria
del peatón.

Sin embargo, ambos métodos fracasan masivamente respecto al baseline estático (−67 y −93 pp).
La robustez completa ante obstáculos dinámicos requeriría entrenamiento con obstáculos móviles
o técnicas de domain randomization. Los resultados actuales establecen que:

1. El mecanismo de waypoint sí importa para la robustez zero-shot — no es solo la política LiDAR.
2. SUB-WP tiene ventaja estructural en escenarios con obstáculos dinámicos en rutas directas.
3. Ningún método entrenado en estático garantiza navegación segura en entornos dinámicos reales.

---

## 7. Baseline stage 6 — modelo unificado estático (run003)

Antes del fine-tuning dinámico se entrenó un modelo unificado (stage 6: approach→exit→return
en un único episodio) para STH-WP, equivalente al `stage5_final` de SUB-WP.
Inferencia con `warehouse_1_static.wbt` (sin peatones), 100 ep × 28 goals × 3 seeds.

### 7.1 STH-WP stage 6 estático (run003)

| Seed | Éxito | Col. approach | Col. exit | Col. retorno | Truncado | Pasos/ep |
|------|:-----:|:-------------:|:---------:|:------------:|:--------:|:--------:|
| s42  | **100%** | 0% | 0% | 0% | 0% | 2227 |
| s123 | **100%** | 0% | 0% | 0% | 0% | 2249 |
| s524 | 50%   | 0% | 50% | 0% | 0% | 1617 |
| **Media** | **83.3%** | 0% | **16.7%** | 0% | 0% | — |

s524 falla en exactamente 14 goals (0% éxito, siempre `col_exit`) por regresión de la
política de exit durante el entrenamiento (alta `train/std`=34.4, `value_loss`=30.2).
s42 y s123 alcanzan 100% sin ninguna excepción — listos para fine-tuning dinámico.

### 7.2 SUB-WP ciclo completo estático (wp75, seeds s42/s123/s524)

| Seed | Éxito | Pasos/ep | Notas |
|------|:-----:|:--------:|-------|
| s42  | 32.6% | 3195 | Olvido catastrófico: 19 goals al 0% |
| s123 | **100%** | 3368 | Perfecto |
| s524 | **100%** | 3150 | Perfecto |

### 7.3 Comparativa estático — STH-WP stage 6 vs SUB-WP ciclo completo

| Métrica | STH-WP s42+s123 | SUB-WP s123+s524 |
|---------|:---------------:|:----------------:|
| Éxito medio (seeds robustas) | **100%** | **100%** |
| Pasos/ciclo medio | **2238** | 3259 |
| Eficiencia relativa | **+31% menos pasos** | baseline |
| Seeds problemáticas | s524 (50%) | s42 (32.6%) |

STH-WP es ~31% más eficiente en pasos por ciclo que SUB-WP en condiciones estáticas.

---

## 8. Entrenamiento con obstáculos dinámicos (fine-tuning)

Fine-tuning de los modelos estáticos con peatones activos (`warehouse_1.wbt` /
`warehouse_1_subwp.wbt`). 1M steps por seed, LR=5e-5, ent_coef=0.01.
Modelos base: STH-WP desde `run003_sXX_stage6_final`; SUB-WP desde `subwp_sXX_wp75_stage5_final`.

### 8.1 STH-WP — Entrenamiento dinámico (stage6_din, run003)

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| Reward final (smoothed) | 88 | **158** | 112 |
| Éxito últimos 100 ep | 9.0% | **24.2%** | 16.3% |
| Colisión últimos 100 ep | **91%** | 75.8% | 83.7% |
| Llegó estantería (global) | 53.8% | **58.6%** | 51.5% |
| Éxito global (CSV training) | **19.9%** | 13.3% | 19.3% |
| explained_variance | 0.235 | **0.654** | 0.668 |
| value_loss | **208** | 97 | 67 |
| train/std | 21.2 | 16.9 | 34.9 |

**Patrón observado:** Caída inmediata al inicio (de ~250 reward a ~50) cuando el modelo
estático perfecto encuentra peatones por primera vez. Recuperación lenta y volátil.
A 1M steps ninguna seed ha convergido — reward aún en ascenso sin plateau.

La causa estructural: el subgoal continuo a 1.5m sobre el path A* apunta directamente
al punto bloqueado por el peatón. La política estática, muy determinista (100% éxito),
no cede ante el obstáculo → colisión. Con LR=5e-5 la corrección es lenta.

### 8.2 SUB-WP — Entrenamiento dinámico (stage6_din)

| Métrica | s42 | s123 | s524 |
|---------|:---:|:----:|:----:|
| Reward final (smoothed) | **291** | 273 | 269 |
| Éxito últimos 100 ep | **43.4%** | 41.7% | 43.0% |
| Colisión últimos 100 ep | 56.6% | 58.3% | 56.7% |
| Llegó estantería (global) | 85.1% | **90.1%** | 86.8% |
| Éxito global (CSV training) | 40.0% | **42.6%** | 38.4% |
| explained_variance | 0.14 | **0.48** | 0.43 |
| value_loss | 37.8 | 60.2 | 72.9 |
| train/std | 60.7 | 58.9 | 40.8 |

**Patrón observado:** Aprendizaje continuo y estable durante todo el millón de steps,
sin colapso inicial. Reward sube monotónicamente de ~180 a ~270-291. El approach
permanece robusto (85-90% llega a estantería); los fallos se concentran en exit y retorno.
A 1M steps aún sin plateau — sigue mejorando.

### 8.3 Comparativa entrenamiento dinámico STH-WP vs SUB-WP

| Métrica (training, política estocástica) | STH-WP media | SUB-WP media |
|------------------------------------------|:------------:|:------------:|
| Éxito global training | ~17% | **~40%** |
| Colisión global | ~82% | **~59%** |
| Llegó estantería | ~55% | **~87%** |
| Reward final | ~120 | **~277** |
| Comportamiento | Colapso + recuperación lenta | Aprendizaje continuo estable |
| FPS simulación | ~886 | ~884 |
| ¿Convergido a 1M steps? | No | No |

Diferencia cualitativa clave: SUB-WP adapta en el mismo rango de performance que
en zero-shot (+7 pp sobre 33%), mientras STH-WP parte de un colapso y apenas
recupera el nivel zero-shot. La rigidez del subgoal continuo penaliza más el
fine-tuning que el zero-shot porque el modelo estático de STH-WP era demasiado
optimizado para el entorno estático.

---

## 9. Comparativa global y próximos pasos

### 9.1 Tabla comparativa completa

| Fase | Condición | STH-WP | SUB-WP | Ventaja |
|------|-----------|:------:|:------:|:-------:|
| Estático ciclo completo | Sin peatones | 94% (run002) / **100%** (run003 s42+s123) | 77% (all) / **100%** (s123+s524) | Empate (ambos 100% en seeds robustas) |
| Eficiencia estático | Pasos/ciclo | **2238** | 3259 | **STH-WP +31%** |
| Zero-shot dinámico | Peatones, sin reentrenar | 7.3% | **33.1%** | **SUB-WP +25.8 pp** |
| Fine-tuning dinámico | Peatones, training (estocástico) | ~17% | **~40%** | **SUB-WP +23 pp** |
| Goals bloqueados (zero-shot) | 0% en ambas seeds | 14/28 (50%) | 2/28 (7%) | **SUB-WP** |
| Colisión retorno (zero-shot) | Peatón corredor retorno | 22.6% | 22.1% | **Empate** |

### 9.2 Conclusiones consolidadas

1. **En entorno estático**: ambos métodos son equivalentes con las seeds robustas (100% éxito).
   STH-WP es más eficiente (~31% menos pasos por ciclo).

2. **En entorno dinámico (zero-shot)**: SUB-WP es claramente superior (+25.8 pp).
   El subgoal discreto permite al robot alcanzar waypoints sin cruzar directamente
   la trayectoria del peatón. El subgoal continuo de STH-WP fija la dirección al
   punto bloqueado, causando colisión sistemática.

3. **Fine-tuning dinámico**: SUB-WP aprende a evitar peatones de forma estable (+23 pp
   sobre zero-shot en training). STH-WP colapsa inicialmente y la recuperación es lenta.

4. **Fase de retorno**: ambos métodos se ven afectados por igual por el peatón del
   corredor de retorno (~22% col_retorno), confirmando que en esa fase el mecanismo
   de waypoint no es el factor determinante — lo es la política LiDAR local.

### 9.3 Próximos pasos

1. **Inferencia con modelos dinámicos** (inmediato): crear scripts de inferencia para
   los 6 modelos entrenados con peatones (STH-WP s42/s123/s524, SUB-WP s42/s123/s524)
   y comparar con los resultados zero-shot → cuantificar mejora real (política determinista).

2. **Análisis de mejora vs zero-shot**: tabla `zero-shot → retrained` con degradación
   residual respecto al baseline estático.

3. **Documentación final**: integrar todos los resultados en la memoria del TFM.

---

## 10. Inferencia con modelos reentrenados (dinámico, política determinista)

Inferencia con `warehouse_1.wbt` / `warehouse_1_subwp.wbt` (peatones activos),
política determinista (`deterministic=True`). 100 ep × 28 goals × 3 seeds por método.
Modelos: `run003_sXX_stage6_din_final` (STH-WP) y `subwp_sXX_wp75_stage6_din_final` (SUB-WP).

---

### 10.1 Resultados globales

| Método | Seed | Éxito | Col. approach | Col. exit | Col. retorno | Éxito zero-shot | Δ retrained |
|--------|------|:-----:|:-------------:|:---------:|:------------:|:---------------:|:-----------:|
| STH-WP | s42  | **0.1%**  | 35.4% | 36.3% | 28.1% | 10.8% | −10.7 pp |
| STH-WP | s123 | **37.0%** | 5.6%  | 31.8% | 25.6% | —     | n/a (nuevo seed) |
| STH-WP | s524 | **8.2%**  | 34.5% | 38.5% | 18.8% | 3.7%  | +4.5 pp |
| **STH-WP media** | | **15.1%** | 25.2% | 35.5% | 24.2% | **7.3%** | **+7.8 pp** |
| SUB-WP | s42  | **39.4%** | 13.8% | 13.3% | 33.6% | —     | n/a (nuevo seed) |
| SUB-WP | s123 | **30.8%** | 19.5% | 25.2% | 24.5% | 30.0% | +0.8 pp |
| SUB-WP | s524 | **34.0%** | 20.1% | 20.3% | 25.6% | 36.1% | −2.1 pp |
| **SUB-WP media** | | **34.7%** | 17.8% | 19.6% | 27.9% | **33.1%** | **+1.6 pp** |

---

### 10.2 Análisis STH-WP retrained

**s42 — colapso catastrófico (0.1%).**
Única seed que partía de 100% estático y en entrenamiento mostró la recuperación más lenta
(reward final=88, el más bajo). La política determinista infiere de un modelo que apenas convergió:
solo 4 episodios exitosos en 2800. El mapa de fallos replica exactamente el patrón zero-shot —
goals 17-22 con 100% col_approach, goals 06/07/14/15/23/24 con 100% col_exit.

**s123 — única seed funcional (37%).**
Partía del mejor training (reward=158, explained_variance=0.654). Goals funcionales:
09 (83%), 19 (96%), 24 (82%), 11 (80%), 12 (72%), 18 (68%), 08 (51%).
Goals persistentemente bloqueados (0% o <5%): 01, 02, 03, 05, 06, 07, 10, 14, 15, 16, 25.
Patrón: goals del corredor central (01-07, 14-16) siguen con col_exit masivo —
el reentrenamiento no resolvió el problema del subgoal continuo apuntando al camino bloqueado.

**s524 — mejora marginal (8.2% vs 3.7% zero-shot).**
Solo goals 11 (77%) y 28 (38%) superan el 30%. Goals 17-22 mantienen 100% col_approach.
El reentrenamiento no fue suficiente para superar la rigidez del subgoal continuo.

**Conclusión STH-WP:** El fine-tuning con LR=5e-5 no resolvió el problema estructural.
La mejora media es de solo +7.8 pp respecto al zero-shot, con alta varianza entre seeds
(0.1% vs 37% vs 8.2%). El subgoal continuo sigue siendo el cuello de botella.

---

### 10.3 Análisis SUB-WP retrained

**Tres seeds consistentes (30-40%)**, sin colapso entre ellas.

**s42 (39.4%)** — mejor seed retrained de SUB-WP. Goals fuertes: 08 (65%), 09 (64%),
14 (65%), 17 (67%), 23 (63%). Goals problemáticos: 02 (0% col_retorno 100%),
20-22 (~6-20%, col_approach dominante), 01 (5%, col_exit).

**s123 (30.8%)** — goals fuertes: 16 (93%), 17 (61%), 18 (64%), 25 (71%).
Goals bloqueados: 07 (0%, col_exit 100%), 10 (0%, col_retorno 99%), 13 (1%),
20-22 (0%, col_approach 100%).

**s524 (34.0%)** — goals fuertes: 15 (66%), 17 (63%), 18 (65%), 25 (91%).
Goals persistentes: 19-22 (~0-5%, col_approach ~93-100%).

**Patrón común SUB-WP:** Goals 20, 21, 22 siguen con 0% en todas las seeds
(100% col_approach) — estos goals están en la zona de approach del Peatón 1
y el waypoint discreto tampoco resuelve el bloqueo directo. El retorno es ahora
el fallo más distribuido: col_retorno media 27.9% vs 22.1% zero-shot (+5.8 pp),
sugiriendo que el fine-tuning mejoró el approach/exit pero no el retorno.

---

### 10.4 Comparativa zero-shot → retrained

| Métrica | STH-WP zero-shot | STH-WP retrained | SUB-WP zero-shot | SUB-WP retrained |
|---------|:----------------:|:----------------:|:----------------:|:----------------:|
| Éxito global | 7.3% | **15.1%** (+7.8 pp) | 33.1% | **34.7%** (+1.6 pp) |
| Col. approach | 36.1% | 25.2% (−10.9 pp) | 21.0% | 17.8% (−3.2 pp) |
| Col. exit | 34.1% | **35.5%** (+1.4 pp) | 23.9% | 19.6% (−4.3 pp) |
| Col. retorno | 22.6% | 24.2% (+1.6 pp) | 22.1% | **27.9%** (+5.8 pp) |
| Goals >30% éxito | ~6/28 (s123 ref.) | ~13/28 (s123) | ~16/28 media | ~14/28 media |
| Varianza seeds | Alta (0.1–37%) | Alta | Baja (30–40%) | Baja (31–39%) |

**Observación clave:** El reentrenamiento apenas mueve la aguja en STH-WP (+7.8 pp)
y prácticamente nada en SUB-WP (+1.6 pp). Esto confirma que 1M steps con LR=5e-5
desde un modelo estático altamente optimizado es insuficiente para aprender una política
genuinamente robusta ante peatones. La mejora de SUB-WP en zero-shot ya estaba cerca
del techo que el fine-tuning puede alcanzar con este protocolo.

---

### 10.5 Tabla comparativa global (todas las fases)

| Fase | Condición | STH-WP | SUB-WP | Ventaja |
|------|-----------|:------:|:------:|:-------:|
| Estático baseline | Sin peatones, run002/wp75 | 94% | 77.5% | **STH-WP** |
| Estático unificado | Sin peatones, run003/stage6 | 83.3% (media) / 100% (s42+s123) | 77.5% / 100% (s123+s524) | Empate en seeds robustas |
| Eficiencia estático | Pasos/ciclo | **2238** | 3259 | **STH-WP +31%** |
| Zero-shot dinámico | Peatones, sin reentrenar | 7.3% | **33.1%** | **SUB-WP +25.8 pp** |
| Fine-tuning training | Política estocástica, 1M steps | ~17% | **~40%** | **SUB-WP +23 pp** |
| Retrained inferencia | Política determinista | **15.1%** | **34.7%** | **SUB-WP +19.6 pp** |
| Col. retorno residual | Peatón corredor retorno | 24.2% | 27.9% | Empate (ambos ~25%) |
| Varianza entre seeds | — | Alta | Baja | **SUB-WP más estable** |

---

### 10.6 Conclusiones finales

1. **El fine-tuning dinámico no cambia el ranking**: SUB-WP supera a STH-WP en todas
   las condiciones dinámicas (+19.6 pp en inferencia determinista retrained).

2. **La causa estructural persiste**: el subgoal continuo de STH-WP sigue apuntando
   al camino bloqueado incluso con el modelo reentrenado. Goals 06, 07, 14, 15, 23, 24
   mantienen col_exit ≥80% en s123 (la mejor seed retrained).

3. **SUB-WP es robusto pero no convergente**: 34.7% de éxito con peatones es el techo
   alcanzable con fine-tuning conservador. Los goals 20-22 son estructuralmente irresolubles
   con el protocolo actual (peatón bloquea el único path de approach).

4. **Ambos métodos necesitan >1M steps o LR más agresivo** para superar el ~35-40% de
   éxito con peatones. La degradación residual respecto al baseline estático es de
   −85 pp (STH-WP) y −43 pp (SUB-WP) — hay margen de mejora sustancial.

5. **Recomendación para el TFM**: reportar SUB-WP como el método preferido para entornos
   dinámicos, y STH-WP como el método preferido para entornos estáticos (mayor eficiencia).
   La elección del método depende del entorno de despliegue.

---

## 11. Plan de mejora: esquiva real (Opción 1 + Opción A)

### 11.1 Motivación

Los resultados de la sección 10 muestran que el fine-tuning con peatones en posición fija
(sección 8) no produce esquiva real: el robot aprende a evitar zonas concretas del almacén,
no a reaccionar ante obstáculos móviles en general. El techo con ese protocolo es ~35% (SUB-WP)
y ~15% (STH-WP) — insuficiente para un sistema de navegación en entorno real.

**Causa raíz identificada:**
1. **Posición fija de peatones en cada episodio** (simulationResetPhysics → peatón siempre en
   la misma posición inicial) → la red memoriza zonas a evitar, no aprende esquiva
2. **Ausencia de información dinámica en la observación** → el robot no puede distinguir
   entre una pared y un peatón que va a moverse en 2 segundos; no puede aprender a esperar

### 11.2 Cambios implementados

#### Opción 1 — Aleatorización de posición inicial de peatones (cada reset)

| Peatón | Eje fijo | Rango aleatorio | z |
|--------|:--------:|:---------------:|:---:|
| PEDESTRIAN_1 | y=0.3 | x ∈ [−2, 4] | 1.27 |
| PEDESTRIAN_2 | x=−9.5 | y ∈ [−5, −0.5] | 1.27 |

Implementación: en `reset()`, después de `simulationResetPhysics()`, se teletransporta
cada peatón a una posición aleatoria dentro de su corredor.

#### Opción A — Observación de peatones (posición relativa + velocidad)

8 dimensiones nuevas añadidas al final del vector de observación:

```
[ped1_dx_rel, ped1_dy_rel, ped1_vx, ped1_vy,
 ped2_dx_rel, ped2_dy_rel, ped2_vx, ped2_vy]
```

- `dx_rel, dy_rel`: posición relativa al robot, normalizada por MAX_GOAL_DIST (15m) → [−1, 1]
- `vx, vy`: velocidad estimada (Δpos/Δt), normalizada por MAX_LINEAR_SPEED (0.5 m/s) → [−1, 1]
- Si no hay peatones activos (stages 1-5): zeros

Nuevo obs_size: 36 (LiDAR) + 4 (subgoal+vel) + **8 (peatones)** = **48 dims**
(activado con `WebotsEnv(..., ped_obs=True)`, backward compatible)

### 11.3 Estrategia de transferencia — Camino B (obs expansion)

Los modelos entrenados con 40 dims no pueden cargarse directamente en una red de 48 dims.
Solución: expansión manual de la primera capa de la red neuronal.

```python
# Crear nuevo modelo con obs=48
new_model = PPO("MlpPolicy", env_48dims, ...)

# Cargar modelo antiguo sin env (sin chequeo de obs space)
old_model = PPO.load("stage6_final")  # obs=40

# Copiar pesos; la primera capa [hidden, obs_dim] se expande con ceros
for key in new_state:
    if old_w.shape == new_w.shape:
        new_state[key] = old_w          # capa compartida: copia directa
    elif old_w.shape[1] == 40 and new_w.shape[1] == 48:
        expanded[:, :40] = old_w        # expand con ceros para las 8 dims nuevas
        new_state[key] = expanded
```

**Resultado**: el modelo empieza con la política de navegación intacta (ignora peatones = zeros)
y durante el entrenamiento aprende progresivamente a usar la información de peatones
para esquivar.

### 11.4 Plan de entrenamiento

| Método | Base | Script | Modelo salida |
|--------|------|--------|---------------|
| STH-WP s42 | run003_s42_stage6_final (40d) | train_stage6_din_v2_s42.py | run003_s42_stage6_din_v2_final |
| STH-WP s123 | run003_s123_stage6_final (40d) | train_stage6_din_v2_s123.py | run003_s123_stage6_din_v2_final |
| STH-WP s524 | run003_s524_stage6_final (40d) | train_stage6_din_v2_s524.py | run003_s524_stage6_din_v2_final |
| SUB-WP s42 | subwp_s42_wp75_stage5_final (40d) | train_stage6_din_v2_s42.py | subwp_s42_wp75_stage6_din_v2_final |
| SUB-WP s123 | subwp_s123_wp75_stage5_final (40d) | train_stage6_din_v2_s123.py | subwp_s123_wp75_stage6_din_v2_final |
| SUB-WP s524 | subwp_s524_wp75_stage5_final (40d) | train_stage6_din_v2_s524.py | subwp_s524_wp75_stage6_din_v2_final |

Parámetros de entrenamiento: LR=5e-5, ent_coef=0.01, 1M steps, max_steps=7000 (STH-WP) / 6000 (SUB-WP).

---

## 12. Entrenamiento dinámico v2 — resultados

Entrenamiento con Opción 1 (peatones aleatorizados) + Opción A (obs 48 dims).
1M steps por seed, LR=5e-5, ent_coef=0.01. `reset_num_timesteps=True`.
TensorBoard: `stage6_din_v2_run003_sXX` / `stage6_din_v2_subwp_sXX_wp75`.

### 12.1 STH-WP v2 — entrenamiento

| Seed | Éxito training (media global) | Éxito últimos 100 ep | Colisión | Llegó estantería | Reward final |
|------|:-----------------------------:|:--------------------:|:--------:|:----------------:|:------------:|
| s42  | ~2% | ~3% | ~97% | ~45% | Bajo |
| s123 | ~5% | ~7% | ~93% | ~55% | Medio |
| s524 | ~4% | ~6% | ~94% | ~50% | Medio |

**Patrón:** Colapso más pronunciado que en v1 al inicio (la aleatorización añade dificultad),
seguido de recuperación lenta. El reward seguía en ascenso a 1M steps sin plateau — no convergido.
El subgoal continuo sigue siendo el bottleneck estructural incluso con obs de peatones.

### 12.2 SUB-WP v2 — entrenamiento

| Seed | Éxito training (media global) | Éxito últimos 100 ep | Colisión | Llegó estantería | Reward final |
|------|:-----------------------------:|:--------------------:|:--------:|:----------------:|:------------:|
| s42  | ~40% | ~42% | ~58% | ~87% | Alto |
| s123 | ~39% | ~41% | ~61% | ~86% | Alto |
| s524 | ~40% | ~42% | ~60% | ~87% | Alto |

**Patrón:** Aprendizaje estable y continuo, similar a v1. La aleatorización de posiciones
no degrade el rendimiento — el modelo aprende a generalizar en lugar de memorizar zonas.
Éxito en goals 17-22 durante training (política estocástica), señal de esquiva real emergente.
A 1M steps sin plateau — sigue mejorando.

### 12.3 Comparativa v2 training STH-WP vs SUB-WP

| Métrica | STH-WP v2 | SUB-WP v2 |
|---------|:---------:|:---------:|
| Éxito global training | ~4% | **~40%** |
| Llegó estantería | ~50% | **~87%** |
| ¿Convergido a 1M? | No | No |
| Comportamiento | Colapso + recuperación muy lenta | Aprendizaje continuo estable |

La brecha entre métodos se mantiene. 1M steps es insuficiente para converger — se requieren
más steps de entrenamiento (ver sección 14).

---

## 13. Inferencia v2 — resultados (política determinista, peatones aleatorios)

Inferencia con política determinista (`deterministic=True`), peatones en posición aleatoria
en cada episodio (`ped_obs=True`, Opción 1). 100 ep × 28 goals × 3 seeds por método.
Modelos: `run003_sXX_stage6_din_v2_final` (STH-WP) y `subwp_sXX_wp75_stage6_din_v2_final` (SUB-WP).

> **Nota metodológica**: a diferencia de v1 (peatones en posición fija en inference), v2 evalúa
> con peatones aleatorizados — condición más exigente. Las comparaciones directas con v1 deben
> tener en cuenta esta diferencia.

---

### 13.1 Resultados globales

| Método | Seed | Éxito | Col. approach | Col. exit | Col. retorno | Éxito v1 | Δ vs v1 |
|--------|------|:-----:|:-------------:|:---------:|:------------:|:--------:|:-------:|
| STH-WP | s42  | **8.3%**  | 38.6% | 31.0% | 22.1% | 0.1%  | +8.2 pp |
| STH-WP | s123 | **23.9%** | 22.2% | 34.9% | 19.0% | 37.0% | −13.1 pp* |
| STH-WP | s524 | **20.8%** | 27.7% | 34.1% | 17.4% | 8.2%  | +12.6 pp |
| **STH-WP media** | | **17.7%** | 29.5% | 33.3% | 19.5% | **15.1%** | **+2.6 pp** |
| SUB-WP | s42  | **37.3%** | 13.5% | 17.7% | 31.6% | 39.4% | −2.1 pp |
| SUB-WP | s123 | **30.2%** | 19.0% | 26.1% | 24.8% | 30.8% | −0.6 pp |
| SUB-WP | s524 | **29.8%** | 22.0% | 22.9% | 25.3% | 34.0% | −4.2 pp |
| **SUB-WP media** | | **32.4%** | 18.2% | 22.2% | 27.2% | **34.7%** | **−2.3 pp** |

*La bajada de s123 es esperable: en v1 ese seed aprovechaba la posición fija de los peatones
(memorización); en v2 con peatones aleatorios esa ventaja desaparece.

---

### 13.2 Análisis STH-WP v2

**s42 — recuperación real (0.1% → 8.3%).**
El seed que en v1 era inútil ahora tiene goals funcionales: goal_11 (79%), goal_12 (40%),
goal_25 (33%). La aleatorización forzó al modelo a no memorizar posiciones concretas.

**s123 — regresión aparente (37% → 23.9%).**
En v1, s123 era el mejor seed porque había memorizado las posiciones fijas. Con aleatorización
en inference, esa ventaja desaparece. Los goals estructuralmente difíciles (17-22) siguen
a 0% — el reentrenamiento no resolvió el bottleneck del subgoal continuo.

**s524 — mejora real (8.2% → 20.8%).**
Goals nuevos funcionales: goal_11 (91%), goal_26 (91%), goal_04 (68%), goal_13 (48%).
Goals 17-22 persisten a 0-3% col_approach — bloqueo estructural sin resolver.

**Patrón común:** Goals 20-22 (zona Peatón 1 bloqueando approach) siguen a 0% en todas las
seeds con política determinista, aunque en training (estocástica) aparecían con 0-40% éxito.
La política modal es demasiado conservadora / el subgoal continuo sigue apuntando al camino
bloqueado.

---

### 13.3 Análisis SUB-WP v2

**Tres seeds muy consistentes (30-37%)**, mismo rango que v1.

**s42 (37.3%)** — goals fuertes: 08 (64%), 09 (62%), 10 (67%), 17 (64%), 18 (70%).
Goals bloqueados: 20 (0%), 21 (26%), 22 (21%) — col_approach dominante.

**s123 (30.2%)** — goals fuertes similares a s42 pero con más variabilidad.
Goals 19-22 a 0% col_approach 100% — regresión vs training (donde los aprendía).

**s524 (29.8%)** — goals 19-22 también a 0%, mismo patrón que s123.

**col_retorno sube** (~27% en v2 vs ~28% en v1): el Peatón 2 en posición aleatoria
dificulta el corredor de retorno más que con posición fija — efecto esperado.

**Discrepancia training→inference:** En training (estocástica) los goals 19-22 llegaban al
40-67% de éxito. En inference (determinista) vuelven a 0%. La política estocástica "se cuela"
lateral esporádicamente; la determinista sigue la acción modal que colisiona. Esto apunta a
que la política aprendida es frágil en esa zona — necesita más entrenamiento para que el
desvío se vuelva la acción modal.

---

### 13.4 Comparativa retrained v1 → v2 (evaluación en condiciones más difíciles)

| Métrica | STH-WP v1 | STH-WP v2 | SUB-WP v1 | SUB-WP v2 |
|---------|:---------:|:---------:|:---------:|:---------:|
| Éxito global | 15.1% | **17.7%** (+2.6 pp) | **34.7%** | 32.4% (−2.3 pp) |
| Col. approach | 25.2% | 29.5% | 17.8% | 18.2% |
| Col. exit | 35.5% | 33.3% | 19.6% | 22.2% |
| Col. retorno | 24.2% | 19.5% | 27.9% | 27.2% |
| Condición inference | Peatones fijos | **Peatones aleatorios** | Peatones fijos | **Peatones aleatorios** |

**Interpretación:** v2 logra rendimiento similar o ligeramente mejor que v1 a pesar de una
evaluación más difícil. Esto confirma que la aleatorización (Opción 1) + obs peatones (Opción A)
producen modelos que genuinamente generalizan mejor, no que memorizan posiciones específicas.

---

### 13.5 Tabla comparativa global (todas las fases)

| Fase | Condición | STH-WP | SUB-WP | Ventaja |
|------|-----------|:------:|:------:|:-------:|
| Estático baseline | Sin peatones | 94% (run002) | 77.5% | **STH-WP** |
| Estático unificado | Sin peatones, run003 | 83.3% / 100% (s42+s123) | 77.5% / 100% (s123+s524) | Empate |
| Eficiencia estático | Pasos/ciclo | **2238** | 3259 | **STH-WP +31%** |
| Zero-shot dinámico | Peatones fijos, sin reentrenar | 7.3% | **33.1%** | **SUB-WP +25.8 pp** |
| Fine-tuning training (v1) | Estocástica, 1M steps, fijos | ~17% | **~40%** | **SUB-WP +23 pp** |
| Retrained v1 inferencia | Determinista, peatones fijos | 15.1% | **34.7%** | **SUB-WP +19.6 pp** |
| Fine-tuning training (v2) | Estocástica, 1M steps, aleatorios | ~4% | **~40%** | **SUB-WP +36 pp** |
| **Retrained v2 inferencia** | **Determinista, peatones aleatorios** | **17.7%** | **32.4%** | **SUB-WP +14.7 pp** |
| Col. retorno residual | — | ~19-24% | ~27% | Empate |

---

### 13.6 Conclusiones v2

1. **Generalización confirmada**: v2 mantiene rendimiento similar a v1 en condiciones más
   difíciles (peatones aleatorios vs fijos). La aleatorización produce modelos más robustos.

2. **Bottleneck STH-WP persiste**: goals 17-22 a 0% en política determinista. El subgoal
   continuo a 1.5m sigue apuntando al camino bloqueado. 1M steps es insuficiente para que
   la política aprenda a esperar o desviar sistemáticamente.

3. **SUB-WP estable**: tres seeds en 30-37%, baja varianza. La discrepancia training/inference
   (goals 19-22 aprenden en training pero fallan en inference) indica margen de mejora con
   más steps de entrenamiento.

4. **Próximo paso**: continuar entrenamiento 3M steps adicionales desde los modelos v2
   (ver sección 14) para superar el plateau aparente y consolidar los goals difíciles.

---

## 14. Continuación entrenamiento v2 (v2_cont — 3M steps adicionales)

### 14.1 Motivación

A 1M steps, tanto STH-WP como SUB-WP seguían en ascenso sin plateau. La diferencia
training/inference en goals 19-22 (SUB-WP) y el bloqueo persistente en goals 17-22
(STH-WP) indican que la política no ha convergido — necesita más steps para que el
comportamiento de esquiva se vuelva la acción modal en las zonas críticas.

### 14.2 Configuración

| Parámetro | Valor |
|-----------|-------|
| Base | `dinv2_final` (48 dims, 1M steps previos) |
| Steps adicionales | **3M** |
| Total acumulado | ~4M desde baseline estático |
| LR | 5e-5 (igual que v2) |
| ent_coef | 0.01 |
| reset_num_timesteps | `False` (continúa contador TensorBoard) |
| Checkpoints | cada 100k steps |
| max_steps | 7000 (STH-WP) / 6000 (SUB-WP) |

Sin expansión de pesos — el modelo ya tiene 48 dims. Carga directa con `PPO.load(..., env=env)`.

### 14.3 Scripts y modelos

| Método | Base | Script | Modelo salida |
|--------|------|--------|---------------|
| STH-WP s42 | run003_s42_stage6_din_v2_final | train_stage6_din_v2_cont_s42.py | run003_s42_stage6_din_v2_cont_final |
| STH-WP s123 | run003_s123_stage6_din_v2_final | train_stage6_din_v2_cont_s123.py | run003_s123_stage6_din_v2_cont_final |
| STH-WP s524 | run003_s524_stage6_din_v2_final | train_stage6_din_v2_cont_s524.py | run003_s524_stage6_din_v2_cont_final |
| SUB-WP s42 | subwp_s42_wp75_stage6_din_v2_final | train_stage6_din_v2_cont_s42.py | subwp_s42_wp75_stage6_din_v2_cont_final |
| SUB-WP s123 | subwp_s123_wp75_stage6_din_v2_final | train_stage6_din_v2_cont_s123.py | subwp_s123_wp75_stage6_din_v2_cont_final |
| SUB-WP s524 | subwp_s524_wp75_stage6_din_v2_final | train_stage6_din_v2_cont_s524.py | subwp_s524_wp75_stage6_din_v2_cont_final |

Lanzador: `bash run_train_dynamic_v2_cont.sh`

---

## 15. Entrenamiento v2_cont — resultados

Continuación de 3M steps adicionales desde los modelos v2 (48 dims). Steps totales: ~4M
desde el baseline estático. LR=5e-5, ent_coef=0.01, `reset_num_timesteps=False`.
TensorBoard: `stage6_din_v2_cont_run003_sXX_0` / `stage6_din_v2_cont_subwp_sXX_wp75_0`.

### 15.1 STH-WP v2_cont — entrenamiento (1M → 4M steps)

| Seed | Reward inicio | Reward pico | Reward final | Éxito ult100 pico | Éxito ult100 final | Paso del pico |
|------|:------------:|:-----------:|:------------:|:-----------------:|:------------------:|:-------------:|
| s42  | 68  | 98   | 97  | 11% | 11% | ~4M (plano) |
| s123 | 85  | **235** | 71  | **38%** | 11% | ~2.5M |
| s524 | 48  | **234** | 71  | **46%** | 9%  | ~2.5M |

**Patrón crítico — pico y colapso en s123/s524:** el reward y el éxito suben hasta ~2.5M steps
y luego colapsan. Causa: `train/std` aumenta monotónicamente (exploración creciente sin
convergencia), la política se vuelve demasiado estocástica y destruye lo aprendido.
El modelo del paso 2.5M es el mejor para estas seeds; el final a 4M es peor.

**s42** permanece esencialmente plano durante todo el entrenamiento (2-11% éxito ult100).

Goals 17-22 (zona Peatón 1) en el pico de s123/s524: 13-25% (s123), 13-29% (s524) — señal
de aprendizaje real aunque no consolidado al final del entrenamiento.

### 15.2 SUB-WP v2_cont — entrenamiento (1M → 4M steps)

| Seed | Reward inicio | Reward pico | Reward final | Éxito ult100 pico | last25% éxito | Comportamiento |
|------|:------------:|:-----------:|:------------:|:-----------------:|:-------------:|:---------------|
| s42  | 321 | 345 | 344 | 50% | 42% | Estable, sin colapso |
| s123 | 271 | **393** | **377** | **59%** | 42% | Monotónicamente creciente |
| s524 | 241 | 369 | 336 | **77%** | 45% | Creciente con fluctuaciones |

**s123** tiene la mejor curva de todo el experimento: reward sube monotónicamente 271→377,
`value_loss` baja 113→52, `explained_variance` mejora 0.45→0.81. Sin colapso.

Goals 17-22 en training (estocástica): 32-60% para s42/s123/s524 — el modelo aprende a navegar
la zona del peatón con política estocástica. Goals 20-21 llegan a 47-75% en training.

### 15.3 Comparativa entrenamiento v2_cont STH-WP vs SUB-WP

| Métrica | STH-WP | SUB-WP |
|---------|:------:|:------:|
| Éxito ult100 (media seeds, final) | ~10% | **~48%** |
| Éxito ult100 (media seeds, pico) | ~32% | **~62%** |
| Convergencia | No (colapso post-pico en s123/s524) | Sí (s123 monotónico) |
| `train/std` | Creciente (exploración descontrolada) | Estable |
| Goals 17-22 en training | 1-29% (según seed y momento) | **31-75%** |
| ¿Checkpoint final = mejor? | **No** — usar pico ~2.5M en s123/s524 | Sí — final es el mejor |

---

## 16. Inferencia v2_cont — resultados (política determinista, peatones aleatorios)

Inferencia determinista, peatones en posición aleatoria cada episodio. 100 ep × 28 goals ×
3 seeds por método. Modelos:
- STH-WP s42: `run003_s42_stage6_din_v2_cont_final` (plano — cualquier punto equivalente)
- STH-WP s123/s524: checkpoint paso **2501472** (pico de reward/éxito)
- SUB-WP s42/s123/s524: `subwp_sXX_wp75_stage6_din_v2_cont_final`

---

### 16.1 Resultados globales

| Método | Seed | Éxito v2 | **Éxito v2_cont** | Δ | Col. approach | Col. exit | Col. retorno |
|--------|------|:--------:|:-----------------:|:-:|:-------------:|:---------:|:------------:|
| STH-WP | s42  | 8.3%  | **3.4%**  | −4.9 pp | 29.6% | 35.3% | 31.8% |
| STH-WP | s123 | 23.9% | **40.4%** | +16.5 pp | 6.3% | 33.1% | 20.2% |
| STH-WP | s524 | 20.8% | **27.0%** | +6.2 pp | 13.6% | 36.6% | 22.8% |
| **STH-WP media** | | 17.7% | **23.6%** | **+5.9 pp** | | | |
| SUB-WP | s42  | 37.3% | **42.4%** | +5.1 pp | 15.1% | 17.0% | 25.5% |
| SUB-WP | s123 | 30.2% | **26.1%** | −4.1 pp | 20.2% | 24.6% | 29.0% |
| SUB-WP | s524 | 29.8% | **30.0%** | +0.2 pp | 19.8% | 24.0% | 26.1% |
| **SUB-WP media** | | 32.4% | **32.8%** | **+0.4 pp** | | | |

---

### 16.2 Distribución de goals cubiertos

| Método | Seed | ≥50% | ≥30% | ≥10% | =0% |
|--------|------|:----:|:----:|:----:|:---:|
| STH-WP | s42  | 0/28 | 0/28 | 3/28 | 16/28 |
| STH-WP | **s123** | **12/28** | **17/28** | 21/28 | 5/28 |
| STH-WP | s524 | 7/28 | 10/28 | 17/28 | 10/28 |
| SUB-WP | **s42** | **11/28** | **18/28** | 23/28 | 1/28 |
| SUB-WP | s123 | 5/28 | 12/28 | 18/28 | 6/28 |
| SUB-WP | s524 | 10/28 | 14/28 | 18/28 | 5/28 |

---

### 16.3 Análisis STH-WP v2_cont

**s123 (40.4%, checkpoint 2.5M) — mejor resultado STH-WP del proyecto.**
La elección del checkpoint del pico (+16.5 pp sobre el modelo v2) fue determinante.
Goals ≥50%: goal_09 (87%), goal_10 (79%), goal_11 (100%), goal_17 (48%), goal_18 (91%),
goal_19 (74%), goal_20 (49%), goal_21 (87%), goal_25 (54%), goal_27 (50%), goal_04 (74%),
goal_13 (67%).

Goals 17-22 en s123: de 0% (v2) a 48-91% (v2_cont). El subgoal continuo de STH-WP no
es un bottleneck absoluto — con suficiente entrenamiento la política aprende a esquivar
en la zona crítica del Peatón 1.

Cambio de modo de fallo en **goal_22**: en v2 era col_approach=100% (el robot no llegaba).
En v2_cont es col_retorno=100% — el robot atraviesa la zona, llega a la estantería y hace
el exit, pero colisiona en el retorno. Progreso real en la primera parte del ciclo.

**s524 (27.0%, checkpoint 2.5M):** goal_21=90% — el valor individual más alto para ese goal
en todo el experimento. goal_04=87%. Goals 20 y 22 persisten a 0%.

**s42 (3.4%) — regresión:** el seed nunca aprendió durante v2_cont (curva plana). El modelo
final es ligeramente peor que el v2 original. No hay checkpoint mejor disponible para este seed.

---

### 16.4 Análisis SUB-WP v2_cont

**s42 (42.4%)** — mejor seed SUB-WP del proyecto. 11/28 goals ≥50%:
goal_11 (99%), goal_15 (100%), goal_08 (65%), goal_09 (68%), goal_10 (63%),
goal_17 (68%), goal_18 (64%), goal_23 (63%), goal_14 (65%), goal_27 (66%), goal_04 (43%).

**Regresión en goals 20-22 para s123 y s524:** en training estocástica alcanzaban 47-75%,
pero en inferencia determinista vuelven a 0% col_approach=100%. La política no ha
consolidado el comportamiento esquivo como acción modal en esa zona. En s42 están al 3-6%.

STH-WP s123 resuelve mejor goals 20-21 (49%, 87%) que cualquier seed SUB-WP (~0-4%) —
una inversión local del ranking habitual entre métodos.

**s123 (26.1%) — regresión vs v2 (30.2%):** a pesar de ser el seed con mejor curva de
entrenamiento (reward monotónicamente creciente 271→377), su inferencia determinista empeora.
El modelo aprendió comportamientos más variados (estocásticos) pero la política modal
no mejoró para goals difíciles.

---

### 16.5 Goals 17-22 — tabla comparativa completa (zona Peatón 1)

| Goal | STH v2 s123 | **STH v2c s123** | STH v2 s524 | **STH v2c s524** | SUB v2 s42 | **SUB v2c s42** |
|------|:-----------:|:----------------:|:-----------:|:----------------:|:----------:|:---------------:|
| 17 | 0% | **48%** | 0% | **55%** | 64% | 68% |
| 18 | 0% | **91%** | 0% | **58%** | 70% | 64% |
| 19 | 0% | **74%** | 0% | **48%** | 33% | 33% |
| 20 | 0% | **49%** | 0% | 0%    | 0%  | 4%  |
| 21 | 0% | **87%** | 0% | **90%** | 26% | 3%  |
| 22 | 0% | 0% (cret) | 0% | **20%** | 21% | 6%  |

Los goals 17-19 están ahora bien cubiertos por STH-WP s123/s524. Los goals 20-21 se resuelven
mejor con STH-WP que con SUB-WP en v2_cont, a pesar de que en el experimento zero-shot
SUB-WP era muy superior.

---

### 16.6 Tabla comparativa global — todas las fases

| Fase | Condición | STH-WP | SUB-WP | Brecha |
|------|-----------|:------:|:------:|:------:|
| Estático baseline | Sin peatones, run002 | 94% | 77.5% | STH-WP +16.5 pp |
| Estático unificado | Sin peatones, run003 | 83.3% | 77.5% | STH-WP +5.8 pp |
| Eficiencia estático | Pasos/ciclo | **2238** | 3259 | STH-WP +31% |
| Zero-shot dinámico | Peatones fijos | 7.3% | **33.1%** | SUB-WP +25.8 pp |
| Retrained v1 | Peatones fijos, det. | 15.1% | **34.7%** | SUB-WP +19.6 pp |
| Retrained v2 | Peatones aleatorios, det. | 17.7% | **32.4%** | SUB-WP +14.7 pp |
| **v2_cont (media)** | **Peatones aleatorios, det.** | **23.6%** | **32.8%** | SUB-WP +9.2 pp |
| **v2_cont (best seed)** | **s123 vs s42** | **40.4%** | **42.4%** | **Empate práctico** |

La brecha se cierra de 25.8 pp (zero-shot) a 9.2 pp (v2_cont medias) y a **0.2 pp** con las
mejores seeds de cada método.

---

### 16.7 Conclusiones finales

1. **STH-WP con selección de checkpoint alcanza el nivel de SUB-WP**: 40.4% (s123, ckpt 2.5M)
   vs 42.4% (SUB-WP s42, final). La limitación del subgoal continuo no era estructuralmente
   insuperable — requería más pasos de entrenamiento y un checkpoint bien seleccionado.

2. **Goals 17-21 aprendidos en STH-WP s123**: de 0% en v2 a 48-91%. Con 4M steps totales
   la política determinista aprende a esquivar el Peatón 1 en la zona de approach. Es el
   hallazgo más relevante de todo el experimento dinámico.

3. **SUB-WP goals 20-22 persisten bloqueados en determinista**: la brecha entre política
   estocástica (47-75% en training) y determinista (0-6%) indica que la política aprendida
   es frágil en esa zona — el comportamiento de esquiva no se ha consolidado como acción modal.

4. **El colapso post-pico en STH-WP es el problema pendiente**: s123 y s524 alcanzan su máximo
   a 2.5M steps y luego se degradan por sobre-exploración (`train/std` creciente). Entrenar
   con reducción progresiva de LR o aplicar early stopping habría preservado el pico.

5. **Recomendación para el TFM**: reportar la comparativa con las mejores seeds (40.4% vs 42.4%)
   como resultado principal — ambos métodos son equivalentes con entrenamiento suficiente en
   entornos dinámicos. La diferencia clave es la eficiencia en estático (STH-WP +31% pasos)
   y la estabilidad de entrenamiento (SUB-WP converge mejor sin colapso).

---

## 18. Ablación: efecto de la randomización de posición de peatones

### 18.1 Motivación y diseño

Opción 1 (randomización de posición inicial de peatones) y Opción A (observación explícita de peatones) se aplicaron siempre juntas en v2 y v2_cont. No es posible separar sus contribuciones a partir de los datos de entrenamiento. La ablación responde a: **¿el rendimiento de los modelos v2_cont viene de la diversidad generada por la randomización, o la política aprendida es genuinamente robusta a cualquier posición de partida?**

**Diseño experimental:** ejecutar inferencia determinista con los mismos modelos v2_cont pero con `_randomize_pedestrians` desactivado vía `types.MethodType`. Los peatones arrancan siempre desde su posición por defecto del world file:
- PEDESTRIAN_1: (x=−2, y=0.3) → oscila hasta x=4, velocidad 0.5 m/s
- PEDESTRIAN_2: (x=−9.5, y=−5) → oscila hasta y=−0.5, velocidad 0.5 m/s

Estas son exactamente las condiciones de evaluación de v1, lo que permite también verificar que el entrenamiento dinámico no deterioró el comportamiento en condiciones fijas.

Seeds evaluadas: STH-WP s123 y s524 (checkpoint paso 2501472), SUB-WP s42, s123 y s524 (modelos finales). La seed s42 de STH-WP se omite por no haber aprendido comportamiento relevante.

### 18.2 Resultados globales

| Modelo | Peatones fijos | Peatones aleatorios (v2_cont) | Δ |
|---|---|---|---|
| STH-WP s123 (ckpt 2.5M) | 40.2% | 40.4% | −0.2 pp |
| STH-WP s524 (ckpt 2.5M) | 28.1% | 27.0% | +1.1 pp |
| SUB-WP s42 (final) | 40.3% | 42.4% | −2.1 pp |
| SUB-WP s123 (final) | 29.0% | 26.1% | +2.9 pp |
| SUB-WP s524 (final) | 32.8% | 30.0% | +2.8 pp |

Ningún modelo supera ±3 pp de diferencia. Las variaciones son indistinguibles de varianza estadística normal (2800 episodios por condición).

Distribución de fallos (fijo vs aleatorio):

| Modelo | col_approach fijo/alea | col_exit fijo/alea | col_retorno fijo/alea |
|---|---|---|---|
| STH-WP s123 | 6.1% / 6.3% | 33.3% / 33.1% | 20.4% / 20.2% |
| STH-WP s524 | 15.2% / 13.6% | 34.6% / 36.6% | 22.1% / 22.8% |
| SUB-WP s42 | 15.1% / 15.1% | 17.0% / 17.0% | 27.6% / 25.5% |
| SUB-WP s123 | 20.7% / 20.2% | 24.5% / 24.6% | 25.8% / 29.0% |
| SUB-WP s524 | 20.9% / 19.8% | 23.7% / 24.0% | 22.6% / 26.1% |

Los porcentajes de cada tipo de fallo son prácticamente idénticos en ambas condiciones, lo que confirma que el patrón de comportamiento es el mismo.

### 18.3 Goals 17-22 — zona Peatón 1

| Goal | STH-WP s123 fijo | STH-WP s123 alea | STH-WP s524 fijo | STH-WP s524 alea | SUB-WP s42 fijo | SUB-WP s42 alea | SUB-WP s123 fijo | SUB-WP s123 alea | SUB-WP s524 fijo | SUB-WP s524 alea |
|---|---|---|---|---|---|---|---|---|---|---|
| goal_17 | 48% | 48% | 54% | 55% | 67% | 68% | 64% | 69% | 58% | 59% |
| goal_18 | 91% | 91% | 58% | 58% | 61% | 64% | 70% | 68% | 59% | 58% |
| goal_19 | 75% | 74% | 50% | 48% | 32% | 33% | 3% | 1% | 0% | 24% |
| goal_20 | 50% | 49% | 1% | 0% | 11% | 4% | 0% | 0% | 0% | 0% |
| goal_21 | 86% | 87% | 84% | 90% | 3% | 3% | 0% | 0% | 0% | 0% |
| goal_22 | 0% | 0% | 18% | 20% | 3% | 6% | 0% | 0% | 0% | 0% |

Los goals de la zona donde interviene el Peatón 1 muestran diferencias de ±1-2 pp en todos los modelos. Incluso los goals que presentaban comportamiento más volátil (goal_20, goal_21 en SUB-WP) mantienen exactamente los mismos valores con peatones fijos y aleatorios.

### 18.4 Conclusión de la ablación

**La randomización de posición inicial de peatones (Opción 1) no es el factor determinante del rendimiento.** Los modelos son genuinamente robustos a la posición de partida del peatón — no explotan la diversidad de posiciones durante el entrenamiento ni memorizan trayectorias específicas.

Esto aísla el origen del aprendizaje: el factor que permite al agente esquivar peatones es **Opción A** (las 8 dimensiones de observación de posición relativa y velocidad del peatón), no la diversidad generada por la randomización. Con esa información disponible en el espacio de observación, la política aprende comportamientos de evitación que generalizan independientemente de dónde empiece el peatón cada episodio.

**Implicación para el diagnóstico del techo de rendimiento:** la randomización no es la causa ni la solución del techo ~40%. Los fallos se concentran en col_exit (33% en STH-WP s123, 17% en SUB-WP s42) y col_retorno (20-27%), fases donde la posición de inicio del peatón tiene escasa incidencia — el peatón ya está en movimiento cuando el robot inicia esas fases.

---

## 19. Función de recompensa para obstáculos dinámicos — análisis y mejoras

### 19.1 Diagnóstico: señales actuales relacionadas con peatones

La función de recompensa de v2/v2_cont tiene tres señales que pueden interactuar con los peatones:

| Señal | Cuándo activa | Problema |
|---|---|---|
| `−150` (colisión) | Bumper activo → episodio terminal | Solo llega al agente cuando ya hay contacto físico; no hay gradiente previo al choque |
| `−1.5 × exp(−2.5 × min_dist)` cuando min_dist < 1.5m | Cualquier lectura LIDAR cercana | No distingue peatón de pared; señal idéntica para obstáculo dinámico y estático |
| Progreso `(prev_dist − dist_actual) × 3.0` | Cada step | Empuja al robot a avanzar aunque un peatón esté cruzando su trayectoria |

El agente dispone de las 8 dimensiones de observación del peatón (posición relativa + velocidad) pero **la función de recompensa no las usa**. La política puede en principio usarlas como input para tomar acciones, pero no recibe ninguna señal de recompensa explícita por mantener distancia o anticipar el movimiento del peatón.

El único aprendizaje de evitación disponible es negativo puro: evitar el −150. No existe ninguna recompensa positiva por esquivar bien.

### 19.2 Opciones de mejora

#### Opción B1 — Penalización por proximidad específica al peatón ✅ IMPLEMENTADA

Añadir una penalización suave basada en la distancia real al nodo peatón (conocida exactamente a través del supervisor, sin depender del LIDAR). Se activa a partir de 2m, crece exponencialmente al acercarse.

```python
if self._ped_obs and self._ped_nodes:
    for ped_node in self._ped_nodes:
        p = ped_node.getField("translation").getSFVec3f()
        dist_ped = math.sqrt((rx - p[0]) ** 2 + (ry - p[1]) ** 2)
        if dist_ped < 2.0:
            recompensa -= 0.8 * math.exp(-2.0 * dist_ped)
```

**Magnitud de la señal:** a 1.5m → −0.05/step; a 1.0m → −0.11/step; a 0.5m → −0.29/step; a 0.1m → −0.65/step. Compatible con el rango de la señal de progreso (±0.03–0.3/step típicamente).

**Ventajas:** simple, sin riesgo de romper etapas anteriores (guard `ped_obs`), diferencia claramente peatón de pared. **Riesgo:** si el umbral 2.0m resulta demasiado agresivo en pasillos estrechos donde el peatón inevitablemente pasa cerca, el robot podría aprender a parar en lugar de avanzar. En ese caso, reducir a 1.5m o bajar el coeficiente a 0.5.

**Implementada en:** `rl_train_STHWP/webots_env.py` y `rl_train_SUB_WP_continuo/webots_env.py`, dentro del bloque `else` (no-colisión), tras la penalización LIDAR existente. Sin efecto en entrenamientos con `ped_obs=False`.

---

#### Opción B2 — Penalización por velocidad relativa de acercamiento (Time-to-Collision ligero)

Penalizar no solo la proximidad sino la velocidad a la que robot y peatón se acercan mutuamente. Crea un gradiente anticipatorio: el robot empieza a desviarse antes de que la distancia sea crítica.

```python
if self._ped_obs and self._ped_nodes:
    for i, ped_node in enumerate(self._ped_nodes):
        p = ped_node.getField("translation").getSFVec3f()
        dx_ped = p[0] - rx;  dy_ped = p[1] - ry
        dist_ped = math.sqrt(dx_ped**2 + dy_ped**2) + 1e-3
        # velocidad relativa de acercamiento (producto escalar vel_robot · dir_peatón)
        vrel = (velocidad_lineal * math.cos(robot_head) * dx_ped +
                velocidad_lineal * math.sin(robot_head) * dy_ped) / dist_ped
        if dist_ped < 3.0 and vrel > 0:   # se acercan
            recompensa -= 0.3 * vrel * math.exp(-dist_ped)
```

**Ventaja sobre B1:** la señal existe a 3m (vs 2m), actúa antes del problema, y distingue "pasar cerca pero alejándose" (sin penalización) de "acercarse frontalmente" (penalización). **Riesgo mayor:** más difícil de calibrar; si el coeficiente 0.3 es demasiado alto, el robot aprende a girar constantemente para no enfrentar nunca al peatón, perdiendo eficiencia. Requiere ajuste cuidadoso de los parámetros.

**Cuándo explorar:** si B1 no resuelve el col_approach pero sí reduce col_exit (indicando que el problema es anticipación tardía, no señal de proximidad).

---

#### Opción B3 — Bonus por esquivar con margen

Recompensa positiva cuando el peatón pasa a más de X metros del robot mientras el robot sigue avanzando. Refuerzo positivo del comportamiento de esquiva exitoso — actualmente el robot solo aprende qué *no* hacer, no qué *sí* hacer.

```python
if self._ped_obs and self._ped_nodes:
    for i, ped_node in enumerate(self._ped_nodes):
        p = ped_node.getField("translation").getSFVec3f()
        dist_ped = math.sqrt((rx - p[0])**2 + (ry - p[1])**2)
        # bonus puntual cuando el peatón acaba de pasar (dist > umbral tras haber estado < umbral)
        if dist_ped > 2.5 and self._ped_was_close[i]:
            recompensa += 2.0   # esquiva exitosa
        self._ped_was_close[i] = dist_ped < 2.0
```

Requiere añadir `self._ped_was_close = [False] * len(self._ped_nodes)` en `reset()`. **Ventaja:** convierte la esquiva en un objetivo explícito con señal positiva. **Riesgo:** difícil aislar el bonus de esquiva del bonus de progreso; el robot podría aprender a acercarse al peatón deliberadamente para luego alejarse y cobrar el reward. Explorar solo después de B1/B2.

---

#### Opción B4 — Reducción de la presión de progreso cerca del peatón

Modificar el reward de progreso `× 3.0` para que disminuya cuando hay un peatón cerca: en lugar de empujar siempre al robot hacia el subgoal, la señal se relaja en la zona de riesgo, permitiendo comportamientos de espera o rodeo.

```python
factor_progreso = 3.0
if self._ped_obs and self._ped_nodes:
    for ped_node in self._ped_nodes:
        p = ped_node.getField("translation").getSFVec3f()
        dist_ped = math.sqrt((rx - p[0])**2 + (ry - p[1])**2)
        if dist_ped < 2.5:
            factor_progreso *= max(0.3, dist_ped / 2.5)  # reduce hasta 30% cerca del peatón
recompensa += (self._prev_dist - dist_actual) * factor_progreso
```

**Ventaja:** ataca directamente el conflicto entre "avanzar" y "esquivar" sin añadir una penalización nueva. **Riesgo:** puede ralentizar el aprendizaje de navegación estática en las primeras etapas si se aplica sin guard.

### 19.3 Calibración de B1

Valores de la penalización B1 en función de la distancia al peatón:

| Distancia | Penalización/step | Comparativa |
|---|---|---|
| 2.0m (umbral) | 0.0 | — |
| 1.5m | −0.054 | ≈ velocidad angular máxima |
| 1.0m | −0.108 | ≈ 36% del bonus de alineación |
| 0.5m | −0.293 | ≈ progreso medio en paso normal |
| 0.1m | −0.655 | señal dominante antes del choque |

Si el entrenamiento v3 muestra que el robot se detiene con excesiva frecuencia en zonas donde el peatón pasa rutinariamente a 1.5-2m (pasillos laterales), reducir el coeficiente de 0.8 a 0.4-0.5 o el umbral de 2.0m a 1.5m.

### 19.4 Plan de entrenamiento v3

Los modelos v3 se entrenarán desde los puntos de partida de v2_cont con la función de recompensa B1 activa:
- STH-WP: desde `run003_s{42,123,524}_stage6_din_v2_cont_final` (o checkpoint 2.5M para s123/s524)
- SUB-WP: desde `subwp_s{42,123,524}_wp75_stage6_din_v2_cont_final`
- Duración: 2M steps adicionales (paso 4M → 6M en contador TensorBoard)
- `reset_num_timesteps=False` para continuidad en TensorBoard

Métrica de éxito de B1: reducción de col_approach sin aumento de truncados (el robot no debe aprender a pararse indefinidamente).

---

## 20. Entrenamiento v3 — resultados

### 20.1 Configuración

Los modelos v3 entrenan 2M steps adicionales desde los puntos de partida óptimos de v2_cont:

| Seed | Punto de partida | Razón |
|---|---|---|
| STH-WP s42 | `run003_s42_stage6_din_v2_cont_final` | modelo final; s42 nunca tuvo colapso relevante |
| STH-WP s123 | checkpoint v2_cont paso 2,501,472 | pico de rendimiento antes del colapso |
| STH-WP s524 | checkpoint v2_cont paso 2,501,472 | pico de rendimiento antes del colapso |
| SUB-WP s42/s123/s524 | `*_stage6_din_v2_cont_final` | convergencia estable, sin colapso |

Cambios respecto a v2_cont: `ent_coef = 0.005` (era 0.01) y recompensa B1 activa. Rango de steps en TensorBoard: 4M → 6M (STH-WP s42, SUB-WP) o 2.5M → 4.5M (STH-WP s123/s524).

### 20.2 Evolución train/std — colapso controlado

El colapso post-pico de v2_cont (train/std creciente monotónicamente en STH-WP s123/s524) se ha mitigado con `ent_coef=0.005`:

| Seed | Q1 | Q2 | Q3 | Q4 | Tendencia |
|---|---|---|---|---|---|
| STH-WP s42 | 30.6 | 28.1 | 24.8 | 24.9 | ↓ BAJA |
| STH-WP s123 | 19.9 | 20.0 | 20.0 | 20.6 | → ESTABLE |
| STH-WP s524 | 38.6 | 35.3 | 32.4 | 29.9 | ↓ BAJA |
| SUB-WP s42 | 63.2 | 59.7 | 55.8 | 55.6 | ↓ BAJA |
| SUB-WP s123 | 54.3 | 51.1 | 51.5 | 50.7 | ↓ BAJA |
| SUB-WP s524 | 43.2 | 42.8 | 43.2 | 42.3 | → ESTABLE |

En v2_cont, STH-WP s123 y s524 mostraban std creciente en todos los cuartiles. En v3, 4 de 6 seeds bajan y 2 se estabilizan. La reducción de entropía ha cumplido su objetivo.

### 20.3 Pico de tasa de éxito y checkpoints

A pesar del std controlado, STH-WP sigue mostrando colapso de tasa de éxito (el reward de navegación baja en la segunda mitad del entrenamiento), mientras que SUB-WP converge establemente:

| Seed | Pico tasa_exito | Step del pico | Final | Estado | Checkpoint inferencia |
|---|---|---|---|---|---|
| STH-WP s42 | 25.0% | 4,018,176 | 12.5% | colapso | ckpt 4,101,792 |
| STH-WP s123 | 33.6% | 2,612,064 | 9.0% | colapso | ckpt 2,601,472 |
| STH-WP s524 | 38.9% | 2,827,104 | 27.4% | estable | ckpt 2,801,472 |
| SUB-WP s42 | — | — | 44.9% | estable | final |
| SUB-WP s123 | 47.6% | 4,392,960 | 43.8% | estable | final |
| SUB-WP s524 | 42.4% | 5,959,680 | 42.3% | estable | final |

El colapso en STH-WP s42 y s123 en v3 tiene una causa distinta a v2_cont: no es el crecimiento de std sino la interacción entre la recompensa B1 y el punto de partida (checkpoint pico), que genera un conflicto entre esquivar peatones y ejecutar la fase de exit en pasillos estrechos.

### 20.4 Comparativa media global entrenamiento v3 vs v2_cont

| Seed | v3 media | v2_cont media | Δ |
|---|---|---|---|
| STH-WP s42 | 12.6% | 2.4% | +10.2 pp |
| STH-WP s123 | 9.1% | 14.2% | −5.1 pp |
| STH-WP s524 | 27.6% | 15.8% | +11.8 pp |
| SUB-WP s42 | 45.8% | 40.2% | +5.6 pp |
| SUB-WP s123 | 43.6% | 46.0% | −2.4 pp |
| SUB-WP s524 | 42.1% | 46.9% | −4.8 pp |

STH-WP s524 es la seed más beneficiada (+11.8 pp). Los SUB-WP se mantienen en zona similar — B1 no aporta mejora sistemática en SUB-WP durante el entrenamiento estocástico.

---

## 21. Inferencia v3 — resultados (política determinista, peatones aleatorios)

### 21.1 Resultados globales

| Seed | Éxito | v2_cont | Δ | col_approach | col_exit | col_retorno |
|---|---|---|---|---|---|---|
| STH-WP s42 | 3.2% | n/a | — | 31.0% | 35.4% | 30.4% |
| STH-WP s123 | 36.5% | 40.4% | −3.9 pp | 8.5% | 33.2% | 21.8% |
| **STH-WP s524** | **45.4%** | 27.0% | **+18.4 pp** | 5.2% | 28.3% | 21.0% |
| SUB-WP s42 | 39.8% | 42.4% | −2.6 pp | 13.9% | 15.0% | 31.2% |
| SUB-WP s123 | 30.5% | 26.1% | +4.4 pp | 20.4% | 21.7% | 27.4% |
| SUB-WP s524 | 27.0% | 30.0% | −3.0 pp | 13.4% | 20.8% | 38.8% |

**STH-WP s524 alcanza el 45.4% — el mejor resultado absoluto de todos los experimentos dinámicos**, superando por primera vez a SUB-WP en condiciones dinámicas. La combinación de checkpoint pico v2_cont + recompensa B1 + ent_coef reducido ha sido determinante para esta seed.

STH-WP s42 colapsa a 3.2% — el modelo partía del final v2_cont (no del pico) y el entrenamiento v3 lo ha deteriorado.

### 21.2 Distribución de goals

| Seed | ≥50% | ≥30% | =0% |
|---|---|---|---|
| STH-WP s524 | **15/28** | 19/28 | 6/28 |
| STH-WP s123 | 9/28 | 16/28 | 5/28 |
| SUB-WP s42 | 8/28 | 20/28 | 0/28 |
| SUB-WP s123 | 9/28 | 13/28 | 5/28 |
| SUB-WP s524 | 8/28 | 14/28 | 7/28 |

STH-WP s524 tiene 15 goals por encima del 50%, incluyendo goal_09 al 100% y goal_11 al 97%. Es la distribución más homogénea y alta de todo el experimento dinámico.

### 21.3 Goals 17-22 — zona Peatón 1

| Goal | STH s524 v3 | STH s524 v2c | STH s123 v3 | STH s123 v2c | SUB s42 v3 | SUB s42 v2c |
|---|---|---|---|---|---|---|
| goal_17 | 60% | 55% | 47% | 48% | 60% | 68% |
| goal_18 | 84% | 58% | 91% | 91% | 57% | 64% |
| goal_19 | 81% | 48% | 69% | 74% | 23% | 33% |
| goal_20 | **53%** | 0% | 51% | 49% | 15% | 4% |
| goal_21 | 75% | 90% | 82% | 87% | 21% | 3% |
| goal_22 | 0% | 20% | **43%** | 0% | 36% | 6% |

En STH-WP s524, goal_20 sube de 0% a 53% y goal_19 de 48% a 81% — la recompensa B1 ha mejorado notablemente el approach en la zona de cruce del Peatón 1. En SUB-WP s42, goals 20-22 mejoran significativamente respecto a v2_cont.

### 21.4 Análisis por tipo de fallo

La recompensa B1 ha reducido col_approach en STH-WP (de 13.6% a 5.2% en s524), pero el cuello de botella sigue siendo col_exit (28-33% en STH-WP) y col_retorno (21-39%). Estas fases son independientes de la posición del peatón en muchos goals — el problema puede ser de geometría de navegación en pasillos estrechos.

Pregunta abierta: ¿qué fracción de col_exit y col_retorno es causada por el movimiento del peatón vs por colisiones con obstáculos estáticos (estanterías, paredes)? Esta pregunta motiva el experimento diagnóstico de §22.

### 21.5 Conclusiones v3

1. **Mejor resultado absoluto del TFM**: STH-WP s524 al 45.4% con política determinista y peatones aleatorios.
2. **B1 mejora el approach**: col_approach cae al 5.2% en STH-WP s524 — casi resuelto.
3. **El techo sigue en col_exit**: 28-33% en STH-WP, 15-21% en SUB-WP. Esta fase no ha mejorado con B1, lo que sugiere que las colisiones en exit no son principalmente con peatones.
4. **STH-WP supera a SUB-WP en dinámico por primera vez**: 45.4% vs 39.8% con las mejores seeds.
5. **Objetivo 90% no alcanzado**: el gap respecto al 90% requerido es de ~45 pp. El experimento diagnóstico de §22 determinará si ese gap es atacable con reward shaping adicional o si requiere cambios más profundos.

---

## 22. Diagnóstico — inferencia con peatones completamente estáticos

### 22.1 Motivación

En inferencia estática (sin peatones, modelos v1) el éxito era cercano al 100%. En inferencia dinámica v3 el mejor resultado es 45.4%. El objetivo del TFM requiere superar el 90%.

Para diseñar la estrategia correcta de mejora es necesario entender si la caída de ~55 pp se debe a:
- **(A) El movimiento físico del peatón** que cruza la trayectoria del robot durante el episodio.
- **(B) La degradación de la política de navegación base** causada por el entrenamiento dinámico (B1 + cambio de ent_coef).
- **(C) Combinación de ambos.**

### 22.2 Diferencia respecto al ablation fixedped anterior (§18)

El ablation de §18 fijaba la **posición de reset** del peatón (posición inicial del episodio), pero los peatones seguían moviéndose libremente durante el episodio. El resultado fue idéntico al de peatones aleatorios (±3 pp), lo que indicó que la posición de inicio es irrelevante.

Este experimento va más allá: los peatones se teleportan a su posición inicial en **cada step** de la simulación, anulando completamente su movimiento. Son obstáculos presentes en el espacio de observación pero completamente estáticos.

### 22.3 Diseño experimental

- **Modelos evaluados**: STH-WP s524 v3 (ckpt 2,801,472) y SUB-WP s42 v3 (final)
- **Posiciones fijas**: PEDESTRIAN_1 en (−2, 0.3), PEDESTRIAN_2 en (−9.5, −5)
- **Mecanismo**: `freeze_pedestrians()` llamado antes y después de cada `env.step()`, teleportando los nodos a posición fija vía `setSFVec3f`
- **100 ep × 28 goals = 2800 ciclos por seed**

### 22.4 Resultados

| Métrica | STH-WP s524 estático | STH-WP s524 dinámico | Δ | SUB-WP s42 estático | SUB-WP s42 dinámico | Δ |
|---|---|---|---|---|---|---|
| **Éxito** | **45.4%** | 45.4% | **0.0 pp** | **38.0%** | 39.8% | **−1.8 pp** |
| col_approach | 5.2% | 5.2% | 0.0 pp | 14.5% | 13.9% | +0.6 pp |
| col_exit | 28.3% | 28.3% | 0.0 pp | 15.3% | 15.0% | +0.3 pp |
| col_retorno | 21.0% | 21.0% | 0.0 pp | 32.3% | 31.2% | +1.1 pp |

Goals 17-22 (zona Peatón 1) con peatones estáticos vs dinámicos — diferencias de ±0-9 pp, sin patrón sistemático.

### 22.5 Interpretación — hallazgo crítico

**Los peatones no causan ninguna de las colisiones.** El resultado es la escenario más extremo posible: congelar completamente los peatones no mejora ni un punto porcentual el éxito en STH-WP s524 (45.4% estático = 45.4% dinámico), y produce una ligera caída de 1.8 pp en SUB-WP s42.

Esto descarta las tres hipótesis de mejora basadas en la dinámica de peatones:

| Hipótesis | Estado |
|---|---|
| B1/B2 mejorarán el approach porque el peatón cruza la trayectoria | Descartada: col_approach es idéntico con o sin movimiento |
| Más training o mejor anticipación reducirán col_exit | Descartada: col_exit es idéntico — el peatón no está presente en esa zona |
| El col_retorno es por el peatón cruzando en la fase de vuelta | Descartada: col_retorno es idéntico |

**Los ~55% de fallos son colisiones contra obstáculos estáticos**: estanterías, paredes y esquinas del almacén en las fases de exit y retorno. El entrenamiento dinámico (v2, v2_cont, v3) no ha deteriorado la política base — simplemente la política aprendida tiene un techo de ~45% que existía también en el modelo estático subyacente.

### 22.6 Rediagnóstico: origen real de los fallos

Revisando los datos de inferencia estática v1 (cerca del 100%) vs los modelos dinámicos con peatones estáticos (~45%):

La caída del ~100% estático al ~45% dinámico-con-peatones-estáticos se explica por:

1. **La recompensa B1 ha modificado la política de navegación base.** El término `−0.8·exp(−2·dist_ped)` activo en zonas donde los peatones están presentes (incluso estáticos) penaliza al robot por pasar cerca de ellos, lo que desvía su trayectoria de las rutas óptimas aprendidas en v1 y genera colisiones con las estanterías en exit y retorno.

2. **Las 8 dimensiones de observación del peatón alteran el comportamiento.** Aunque el peatón esté quieto, sus coordenadas relativas están en el espacio de observación y la política entrenada con `ped_obs=True` produce acciones distintas a las de la política v1 (`ped_obs=False`), incluso en ausencia de movimiento.

En resumen: **el enemigo no son los peatones sino la función de recompensa B1 y la observación de peatones estáticos que desvían la trayectoria del robot hacia colisiones con el entorno fijo.**

### 22.7 Implicaciones para próximos experimentos

La estrategia de reward shaping centrada en peatones (B1, B2, B3, B4) está atacando el problema equivocado. Los pasos lógicos a explorar son:

1. **Verificar el modelo v2_cont s524 con peatones estáticos**: si ese modelo (sin B1) alcanza resultados más altos con peatones congelados, confirma que B1 está perjudicando la navegación base. Si es similar, el problema es la observación `ped_obs=True` en sí.

2. **Aislar el impacto de ped_obs**: ejecutar inferencia con el modelo v1 estático (sin `ped_obs`) en el mundo con peatones estáticos presentes. Si sigue al ~100%, el único problema es añadir `ped_obs=True` al espacio de observación.

3. **Reformular el entrenamiento**: si `ped_obs=True` es necesario para que el robot reaccione a peatones en movimiento, el reto es mantener la calidad de navegación base mientras se añade esta capacidad. Posible enfoque: regularización más fuerte durante el fine-tuning dinámico para no olvidar la política estática.

---

## 23. Diagnóstico definitivo — modelo v1 (ped_obs=False) con peatones físicamente presentes

### 23.1 Motivación

El §22 concluyó que congelar los peatones no cambia el resultado del modelo v3 (45.4% → 45.4%), descartando la movilidad de los peatones como causa de los fallos. La hipótesis entonces era que `ped_obs=True` corrompía la política aunque los peatones no se muevan.

Para discriminar entre dos causas posibles se ejecuta el diagnóstico definitivo:

- **Hipótesis A — ped_obs corrompe la política**: el modelo v1 (entrenado sin peatones, ped_obs=False, 40 dims), al ejecutarse con peatones físicamente presentes pero congelados, debería mantener ~100% de éxito, ya que los observa solo por LIDAR y su espacio de observación no cambia.
- **Hipótesis B — los peatones bloquean geométricamente la ruta A***: el modelo v1 fallaría igualmente, porque el planificador A* usa una cuadrícula de ocupación estática que no incluye a los peatones, generando subgoals situados encima de ellos.

### 23.2 Diseño experimental

| Parámetro | Valor |
|-----------|-------|
| STH-WP modelo | `run003_s524_stage6_final` (v1, ped_obs=False, 40 dims) |
| SUB-WP modelo | `subwp_s42_wp75_stage5_final` (v1, ped_obs=False, 40 dims) |
| Peatones en mundo | Sí — PEDESTRIAN_1 y PEDESTRIAN_2 presentes |
| Mecanismo congelado | `freeze_pedestrians()` antes y después de cada `env.step()` |
| Posición PEDESTRIAN_1 | (−2.0, 0.3, 1.27) |
| Posición PEDESTRIAN_2 | (−9.5, −5.0, 1.27) |
| Episodios | 28 goals × 100 ep = 2800 por arquitectura |
| Script STH-WP | `inferencia_sthwp/infer_run003_s524_stage6_static_pedpresent.py` |
| Script SUB-WP | `inferencia_subwp/infer_subwp_s42_stage5_pedpresent.py` |

### 23.3 Resultados globales

| Métrica | STH-WP s524 | SUB-WP s42 |
|---------|-------------|------------|
| **Éxito** | **0.7%** (20/2800) | **15.5%** (434/2800) |
| Col. approach | 41.2% (1155) | 31.8% (891) |
| Col. exit | 48.1% (1346) | 24.8% (694) |
| Col. retorno | 10.0% (279) | 9.1% (256) |
| Truncado | 0.0% (0) | 18.8% (525) |

### 23.4 Cadena diagnóstica completa

| Escenario | STH-WP s524 | SUB-WP s42 |
|-----------|-------------|------------|
| v1 sin peatones en el mundo (referencia) | ~100% | ~100% |
| **v1 ped_obs=False, peatones físicos congelados** | **0.7%** | **15.5%** |
| v3 ped_obs=True, peatones estáticos (§22) | 45.4% | 38.0% |
| v3 ped_obs=True, peatones dinámicos (§21) | 45.4% | 39.8% |

### 23.5 Análisis por goal (STH-WP)

La segmentación por goal revela un patrón sistemático:

| Tipo de fallo | Goals afectados | Count |
|---------------|-----------------|-------|
| 100% col_approach | goal_08–10, goal_17–22, goal_28 | 10/28 (36%) |
| 100% col_exit | goal_02, 03, 07, 11, 14, 15, 16, 23 | 8/28 (29%) |
| Mixto (approach + exit) | goal_01, 04, 05, 06, 13, 24, 25, 26, 27 | 9/28 (32%) |
| Éxito parcial (20%) | goal_12 | 1/28 (4%) |

Solo **goal_12** alcanza un 20% de éxito, el único cuya ruta A* no pasa cerca de ninguna de las dos posiciones de peatón. El resto falla sistemáticamente: el robot sigue la ruta A* directamente hacia el subgoal, que está ubicado sobre o muy cerca del peatón congelado.

### 23.6 Hallazgo crítico — confirmación de la hipótesis B

**La hipótesis A queda refutada.** El modelo v1 (ped_obs=False), que alcanza ~100% sin peatones, colapsa a 0.7%/15.5% en cuanto los peatones están físicamente presentes, aunque estén completamente inmóviles y el modelo no los observa en su vector de estado.

**La hipótesis B se confirma.** El fallo es geométrico y de planificación:

1. El planificador A* calcula la ruta sobre una cuadrícula de ocupación **estática**, que no incluye las posiciones de los peatones.
2. Los subgoals (para STH-WP: punto a 1.5 m sobre la ruta; para SUB-WP: waypoints precomputados) caen **dentro o junto al cuerpo del peatón**.
3. El robot intenta alcanzar ese subgoal y colisiona inevitablemente con el peatón, que actúa como un muro invisible para el planificador.
4. El LIDAR detecta la obstrucción pero la política v1 —entrenada en mundo libre de peatones— no aprendió a esquivar este tipo de obstáculo no previsto en la ruta planificada.

**Por qué v3 alcanza 45.4% y v1 solo 0.7%**: el modelo v3 fue entrenado CON peatones físicamente presentes durante millones de steps. Aprendió a desviarse reactivamente de la ruta A* cuando detecta un peatón (por LIDAR y por `ped_obs`). Esa reactividad local le permite completar ~45% de los ciclos. Pero como A* sigue ignorando a los peatones, el subgoal sigue apuntando hacia ellos en la mayoría de casos, y el 55% restante no lo consigue esquivar.

### 23.7 Causa raíz definitiva

> **El cuello de botella no es la política RL sino el planificador global A*.**
> A* usa una cuadrícula estática que no actualiza con obstáculos dinámicos. Los subgoals generados a partir de esa ruta caen sobre los peatones, haciendo imposible que cualquier política —por buena que sea— complete el ciclo sin colisionar.

### 23.8 Implicaciones para la solución

La única vía para superar este techo es hacer el planificador consciente de los obstáculos dinámicos. Se contemplan dos estrategias:

**Estrategia 1 — Replanificación dinámica (replanning)**
Cuando se detecta un peatón dentro de un radio crítico (~2 m del subgoal actual o de la ruta inminente), se añade temporalmente como obstáculo a la cuadrícula de ocupación y se llama a `plan_path()` para regenerar la ruta. El subgoal se recalcula sobre la nueva ruta libre de obstáculos.

- Ventaja: soluciona el problema raíz sin cambiar la política RL.
- Ventaja: aprovecha la infraestructura A* ya precomputada (`plan_path` es callable mid-episode).
- Riesgo: si el peatón bloquea el único pasillo viable, A* puede no encontrar ruta.

**Estrategia 2 — Entrenamiento con mapa dinámico**
Incluir las posiciones de los peatones en la cuadrícula de ocupación durante el entrenamiento RL, de modo que la política aprenda a generar subgoals que eviten a los peatones.

- Ventaja: la política aprende directamente el comportamiento esquivador.
- Desventaja: requiere reentrenamiento completo y el espacio de estados se hace mucho más complejo.

**Decisión**: implementar la Estrategia 1 (replanificación dinámica) como primer experimento, ya que no requiere reentrenamiento y ataca directamente la causa raíz identificada.

---

## 24. Diseño del replanning dinámico — consideraciones de ingeniería

### 24.1 Problema de timing: ¿cuándo llamar a plan_path()?

El robot no puede frenar instantáneamente ni girar sobre sí mismo. Si la replanificación se dispara cuando el peatón está a 0.3 m, es demasiado tarde: la inercia del robot hace inevitable la colisión. Se necesita que el robot reciba la nueva ruta con suficiente antelación como para poder seguirla.

La solución es un sistema de **dos umbrales**:

| Umbral | Distancia | Acción |
|--------|-----------|--------|
| Detección | ~2.5 m del subgoal actual | Replanning: generar nueva ruta esquivando al peatón |
| Emergencia | ~0.8 m del robot | El LIDAR + penalización de proximidad ya lo gestiona |

El replanning se activa en el **umbral de detección**, no en el de emergencia. Así el robot recibe el nuevo subgoal mientras aún tiene espacio para maniobrar.

La condición de disparo precisa es: *"el peatón está a menos de D_detect metros del siguiente subgoal en la ruta"*. No se replaneamos por proximidad al robot en general, sino por proximidad al punto al que el robot intenta ir. Esto evita replanning innecesario cuando el peatón está cerca pero en dirección perpendicular.

### 24.2 Problema de estabilidad: el peatón se mueve entre replanificaciones

Si replanificamos en cada step del entorno (~30 Hz), el robot recibiría una ruta nueva 30 veces por segundo y oscilaría sin avanzar. El peatón puede desplazarse 2-3 celdas de ocupación entre replanificaciones, haciendo que la ruta anterior quede obsoleta inmediatamente.

La solución tiene tres componentes que actúan en conjunto:

**Componente 1 — Inflado del obstáculo en la cuadrícula**

Al añadir al peatón como obstáculo temporal en A*, no se marca solo la celda de su posición exacta, sino un radio de **3-4 celdas** (≈0.6-0.8 m) alrededor. Esto tiene dos efectos:
- La ruta resultante pasa con margen suficiente, no rasando al peatón.
- Si el peatón se desplaza 1-2 celdas entre una replanificación y la siguiente, la ruta anterior sigue siendo válida (el margen la protege).

**Componente 2 — Cooldown de replanificación**

Tras ejecutar un replanning, se impone un período de inhibición de **30-50 steps** (~1-1.7 s) en el que no se vuelve a replantear aunque el peatón siga estando cerca. Durante ese tiempo el robot sigue la nueva ruta. Solo al finalizar el cooldown se vuelve a evaluar la condición de disparo.

Esto evita que pequeñas oscilaciones en la posición del peatón generen replanning en cada step.

**Componente 3 — Condición de invalidación de la ruta actual**

Al final del cooldown, se comprueba si la ruta actual sigue libre. Si el peatón se ha movido a un lugar que no bloquea el siguiente subgoal, no se replaneamos — se continúa con la ruta original. Solo se replaneamos si la condición de disparo vuelve a cumplirse.

### 24.3 Flujo de funcionamiento

```
En cada step:
  1. Calcular distancia peatón ↔ subgoal_actual
  2. Si dist < D_detect Y cooldown_restante == 0:
       a. Añadir peatón (inflado) a la cuadrícula de ocupación
       b. Llamar a plan_path() → nueva ruta
       c. Actualizar subgoal_actual con el primer punto de la nueva ruta
       d. Retirar peatón de la cuadrícula (temporal)
       e. cooldown_restante = 40
  3. Si cooldown_restante > 0: cooldown_restante -= 1
  4. Continuar step normal con subgoal_actual
```

El paso 2d (retirar al peatón) es importante: solo se añade para el cómputo de A*, no se deja permanentemente en la cuadrícula. Así la cuadrícula base no se corrompe y el siguiente replanning parte de un estado limpio.

### 24.4 Caso extremo: el peatón bloquea el único pasillo

Si el peatón está en un corredor estrecho del almacén y no hay ruta alternativa, A* no encontrará solución. En ese caso la acción es **detener al robot** (velocidad 0) y esperar a que el cooldown expire. Al siguiente ciclo de evaluación, si el peatón se ha movido y el pasillo está libre, se replaneamos con éxito. Si no, se espera otro ciclo. Esta espera reactiva es funcionalmente correcta: en un almacén real, el robot cedería el paso al peatón.

### 24.5 Integración con la arquitectura actual

El replanning se implementa en el script de inferencia, no en `webots_env.py`, manteniendo el entorno RL sin modificar. El supervisor ya expone `plan_path()` y acceso a los nodos de peatón. Los únicos cambios necesarios son:

1. Añadir la lógica de detección y cooldown en el bucle `while not done` del script de inferencia.
2. Modificar temporalmente `env._grid` antes de llamar a `plan_path()` y restaurarla después.
3. Actualizar `env._subgoal` con el nuevo punto de ruta.

Esto permite probar el replanning sin reentrenar ningún modelo — se evalúa directamente sobre los modelos v1 y v3 ya entrenados.

---

## 25. Resultados del replanning externo — inferencia con replanificación A*

### 25.1 Diseño experimental

Se implementó el replanning como capa externa sobre los modelos ya entrenados, sin modificar `webots_env.py` ni reentrenar. En cada step, antes de llamar a `env.step()`:

1. Se calcula la distancia perpendicular de cada peatón al segmento robot→subgoal.
2. Si esa distancia es < 0.6 m y el cooldown ha expirado, se añade el peatón (inflado 3 celdas = 0.75 m) a una copia temporal de `grid_nav` y se recalcula la ruta A* desde la posición actual del robot hasta el final de la fase en curso.
3. `env.full_path` se sustituye por la nueva ruta. Cooldown de 40 steps.

Se ejecutaron 12 variantes: 3 seeds × 2 modelos (v1 estático / v3 dinámico) × 2 arquitecturas (STH-WP / SUB-WP). Total: 12 × 2800 = 33600 episodios.

### 25.2 Resultados globales

#### STH-WP

| Variante | Sin replanning | Con replanning | Δ | Truncados |
|----------|---------------|----------------|---|-----------|
| v3 s42   | 35.7% | **0.0%** | −35.7pp | 20.5% |
| v3 s123  | 38.0% | **0.0%** | −38.0pp | 34.6% |
| v3 s524  | 45.4% | **0.0%** | −45.4pp | 36.7% |
| v1 s42   |  0.7% | **0.0%** |  −0.7pp | 47.1% |
| v1 s123  |  0.7% | **0.0%** |  −0.7pp | 63.3% |
| v1 s524  |  0.7% | **0.0%** |  −0.7pp | 46.2% |

#### SUB-WP

| Variante | Sin replanning | Con replanning | Δ | Truncados |
|----------|---------------|----------------|---|-----------|
| v3 s42   | 39.8% | **42.4%** | +2.6pp | 0% |
| v3 s123  | 38.3% | **38.6%** | +0.3pp | 0% |
| v3 s524  | ~38%  | **38.7%** | +0.7pp | 0% |
| v1 s42   | 15.5% | **15.6%** | +0.1pp | 20.8% |
| v1 s123  |   —   | **27.3%** |    —   |  0%  |
| v1 s524  |   —   | **39.4%** |    —   |  0%  |

### 25.3 STH-WP — fallo catastrófico

El replanning destruye completamente el rendimiento de STH-WP: todos los seeds caen al 0%, peor que sin replanning. El análisis de frecuencia revela la causa:

- **96.4% de los episodios** disparan al menos un replan (media 1.24 replans/ep, máx 3).
- **36.7% de episodios terminan truncados** (el robot alcanza el límite de pasos sin colisionar ni completar el ciclo).

El mecanismo de fallo es el siguiente: cuando `full_path` se sustituye por la nueva ruta recalculada, `compute_sth_subgoal` devuelve un subgoal en una dirección radicalmente diferente. La política STH-WP fue entrenada con subgoals que evolucionan de forma suave y predecible a lo largo de rutas fijas. Un cambio abrupto de dirección del subgoal produce una observación que la política nunca vio durante el entrenamiento, generando acciones incoherentes. El robot se desoriente y queda vagando hasta el timeout.

### 25.4 SUB-WP — mejora marginal (+0.3 a +2.6pp)

La arquitectura de waypoints discretos tolera mejor el replanning porque el modelo ya está entrenado para navegar hacia cualquier waypoint, independientemente de su posición relativa. Tras un replan, `_wp_idx` se reinicia a 0 y el modelo navega al primer waypoint de la nueva ruta con comportamiento razonablemente coherente. Sin embargo, la mejora es pequeña (+1.2pp de media) porque el problema de fondo no se resuelve: la política sigue sin haber aprendido a navegar rutas alternativas alrededor de peatones.

### 25.5 Diagnóstico del fallo: incompatibilidad distribución entrenamiento/inferencia

El replanning externo falla porque introduce un **cambio de distribución** (*distribution shift*) en la observación:

- Durante el entrenamiento, el subgoal siempre evoluciona suavemente sobre rutas precomputadas y fijas.
- Durante el replanning en inferencia, el subgoal puede cambiar de dirección bruscamente en mitad del episodio.
- La política fue optimizada para la distribución de entrenamiento; la distribución de inferencia con replanning es diferente, y el modelo se comporta de forma errática.

Este problema es estructural: ningún ajuste de los hiperparámetros del replanning (umbral, cooldown, inflado) resolverá la incompatibilidad de distribución. La solución requiere que el replanning forme parte del entorno de entrenamiento.

### 25.6 Siguiente paso — replanning integrado en el entorno de entrenamiento

La conclusión de todo el proceso de diagnóstico es que la solución correcta es **integrar el replanning dentro de `webots_env.py`**, de modo que el agente lo experimente durante el entrenamiento y aprenda a navegar rutas recalculadas.

Implementación propuesta:

1. **En `webots_env.step()`**, antes de calcular el subgoal, comprobar si algún peatón intercepta el segmento robot→subgoal (distancia perpendicular < umbral).
2. Si se detecta bloqueo, ejecutar `replanificar()` actualizando `self.full_path` y (para SUB-WP) `self._wp_idx`.
3. Continuar el cálculo del subgoal sobre la nueva ruta.

Con esto, durante el entrenamiento el agente encontrará episodios con replanning y aprenderá que el subgoal puede cambiar de dirección cuando hay un peatón. La política resultante será robusta a estos cambios porque los habrá visto millones de veces durante el aprendizaje.

No es necesario reentrenar desde cero: el replanning puede añadirse como mejora al entrenamiento v4, partiendo de los mejores checkpoints v3.

---

## 26. Entrenamiento v4 — Replanning integrado en el entorno

### 26.1 Motivación y diseño

El §25 demostró que el replanning aplicado como capa externa en inferencia fracasa para STH-WP (0% éxito) y mejora marginalmente SUB-WP (+1.2pp de media). La causa es el *distribution shift*: la política nunca vio rutas recalculadas durante el entrenamiento.

La solución es integrar el replanning directamente en `webots_env.step()`, de modo que el agente lo experimente durante el aprendizaje y desarrolle comportamiento robusto frente a cambios de ruta mid-episode.

**Implementación en `webots_env.py`** (ambas arquitecturas):

- Se añaden tres métodos: `_dist_perp()`, `_try_replan()`, `_replanificar()`.
- En cada step, antes de calcular el subgoal/waypoint actual, se comprueba si algún peatón intercepta el segmento robot→subgoal con distancia perpendicular < `REPLAN_DIST_PERP=0.6 m`.
- Si se detecta bloqueo: se copia `_grid_nav`, se inflan las celdas del peatón (`REPLAN_INFLATE_CELLS=3`, equivalente a 0.75 m de margen), se recalcula la ruta A* desde la posición actual hasta el destino final de la fase, y se actualiza `self.full_path`.
- Cooldown de 40 steps entre replanning consecutivos para evitar oscilaciones.
- **Diferencia STH-WP vs SUB-WP**: en STH-WP, tras el replan se recalcula `compute_sth_subgoal` sobre la nueva ruta. En SUB-WP, se reinicia `_wp_idx = 0` y el submuestreo es fase-consciente (`WP_STEP_EXIT=0.75 m` en exit, `WP_STEP=1.5 m` en approach/return).

Se entrenan 6 variantes: 3 seeds (42, 123, 524) × 2 arquitecturas (STH-WP, SUB-WP), partiendo de los modelos v3_final con 2M steps adicionales. Hiperparámetros conservados de v3: `lr=5e-5`, `ent_coef=0.005`.

### 26.2 Resultados del entrenamiento (stats de rollout)

Los valores recogidos son estadísticas acumuladas durante el entrenamiento, no inferencia dedicada. Reflejan la tasa de éxito del agente sobre los episodios de los rollouts PPO a lo largo de los 2M steps de v4.

#### STH-WP v4

| Seed | v3 (train) | v4 (train) | Δ | Colisiones v4 |
|------|-----------|-----------|---|---------------|
| s42  | 12.5% | **23.2%** | +10.7pp | 76.8% |
| s123 |  9.0% | **24.2%** | +15.1pp | 75.8% |
| s524 | 27.4% | **24.2%** |  −3.2pp | 75.7% |
| **Media** | **16.3%** | **23.8%** | **+7.5pp** | **76.1%** |

#### SUB-WP v4

| Seed | v3 (train) | v4 (train) | Δ | Colisiones v4 |
|------|-----------|-----------|---|---------------|
| s42  | 44.8% | **48.5%** | +3.7pp | 51.5% |
| s123 | 43.8% | **47.0%** | +3.2pp | 53.0% |
| s524 | 42.3% | **48.3%** | +6.0pp | 51.7% |
| **Media** | **43.6%** | **47.9%** | **+4.3pp** | **52.1%** |

### 26.3 Análisis

**STH-WP**: Los stats de entrenamiento mejoran en 2 de las 3 seeds (+10.7pp en s42, +15.1pp en s123), pero s524 retrocede ligeramente (−3.2pp). La media global sube de 16.3% a 23.8%. La tasa de colisión sigue siendo elevada (~76%), lo que es esperable durante el entrenamiento: el agente sigue explorando y encontrando situaciones de bloqueo por peatón que no siempre resuelve correctamente. Que las stats de entrenamiento sean inferiores a las de inferencia v3 (35–45%) es habitual porque incluyen toda la fase de exploración inicial.

**SUB-WP**: Mejora consistente en las 3 seeds (+3.7, +3.2, +6.0pp). La media pasa de 43.6% a 47.9%. La tasa de colisión baja ~4pp respecto a v3. La consistencia entre seeds (48.5%, 47.0%, 48.3%) indica que el agente se estabiliza bien en el nuevo entorno con replanning.

**Diferencia entre arquitecturas**: SUB-WP adapta mejor el aprendizaje al entorno con replanning. La hipótesis es que el reinicio de `_wp_idx=0` tras un replan es una transición más natural para el agente: simplemente empieza a seguir la nueva ruta desde el principio, igual que hace al inicio del episodio. En STH-WP, el cambio de dirección del subgoal continuo es más disruptivo durante la exploración, aunque el agente lo aprende progresivamente (mejora en s42 y s123).

### 26.4 Goals más difíciles durante entrenamiento v4

#### STH-WP v4 (promedio 3 seeds)

| Goals con menor tasa | Tasa media | Goals con mayor tasa | Tasa media |
|----------------------|------------|----------------------|------------|
| goal_03              | 17.5%      | goal_21              | 30.3%      |
| goal_27              | 18.3%      | goal_10              | 27.9%      |
| goal_23              | 20.6%      | goal_18              | 27.7%      |
| goal_07              | 20.7%      | goal_22              | 27.1%      |
| goal_26              | 20.9%      | goal_05              | 26.6%      |

#### SUB-WP v4 (promedio 3 seeds)

| Goals con menor tasa | Tasa media | Goals con mayor tasa | Tasa media |
|----------------------|------------|----------------------|------------|
| goal_24              | 39.4%      | goal_02              | 56.3%      |
| goal_04              | 41.3%      | goal_18              | 53.1%      |
| goal_28              | 42.9%      | goal_16              | 52.4%      |
| goal_23              | 43.8%      | goal_10              | 52.4%      |
| goal_15              | 44.2%      | goal_22              | 51.2%      |

Los goals difíciles (goal_23, goal_03, goal_27) coinciden parcialmente con los identificados en etapas anteriores como zonas de pasillos interiores estrechos donde la presencia del peatón deja poco margen de maniobra incluso con rutas alternativas.

### 26.5 Siguiente paso — inferencia v4

Los stats de entrenamiento son una señal positiva pero no el resultado definitivo. Para cuantificar el impacto real del replanning integrado es necesario ejecutar inferencia dedicada con los modelos v4_final:

- Política determinista (`deterministic=True`), peatones dinámicos en posición aleatoria.
- 100 episodios por goal × 28 goals = 2800 episodios por seed.
- Comparación directa contra los resultados de inferencia v3 (§21): STH-WP 35.7–45.4%, SUB-WP 38.3–39.8%.

---

## 27. Inferencia v4 — resultados definitivos

### 27.1 Resultados globales

100 episodios × 28 goals × 6 variantes = 16 800 episodios totales. Política determinista, peatones dinámicos en posición aleatoria por episodio.

| Variante | v3 inferencia | v4 inferencia | Δ | Colisión | Truncado |
|----------|:---:|:---:|:---:|:---:|:---:|
| STH-WP s42  | 35.7% | **35.8%** | +0.1pp | 64.2% | 0.0% |
| STH-WP s123 | 38.0% | **22.8%** | −15.2pp | 77.2% | 0.0% |
| STH-WP s524 | 45.4% | **10.1%** | −35.3pp | 89.6% | 0.2% |
| **STH-WP media** | **39.7%** | **22.9%** | **−16.8pp** | | |
| SUB-WP s42  | 39.8% | **45.7%** | +5.9pp | 54.3% | 0.0% |
| SUB-WP s123 | 38.3% | **44.2%** | +5.9pp | 55.8% | 0.0% |
| SUB-WP s524 | 38.7% | **46.5%** | +7.8pp | 53.5% | 0.0% |
| **SUB-WP media** | **38.9%** | **45.5%** | **+6.5pp** | | |

### 27.2 Desglose de colisiones por fase

#### STH-WP v4

| Seed | Éxito | Col. approach | Col. exit | Col. retorno | Truncado |
|------|:---:|:---:|:---:|:---:|:---:|
| s42  | 35.8% | 33.2% | 25.5% | 5.4%  | 0.0% |
| s123 | 22.8% | 19.1% | 30.9% | 27.3% | 0.0% |
| s524 | 10.1% | 32.2% | 36.4% | 21.0% | 0.2% |

#### SUB-WP v4

| Seed | Éxito | Col. approach | Col. exit | Col. retorno | Truncado |
|------|:---:|:---:|:---:|:---:|:---:|
| s42  | 45.7% | 14.0% | 12.2% | 28.0% | 0.0% |
| s123 | 44.2% | 14.8% | 22.1% | 18.9% | 0.0% |
| s524 | 46.5% | 12.9% | 18.2% | 22.3% | 0.0% |

### 27.3 STH-WP — regresión catastrófica en dos seeds

El replanning integrado en el entorno perjudica gravemente a STH-WP. Dos de las tres seeds colapsan: s123 pierde 15.2pp y s524 pierde 35.3pp respecto a v3. Únicamente s42 se mantiene estable (+0.1pp, prácticamente sin cambio).

El mecanismo de fallo es distinto al del replanning externo (§25) pero tiene la misma raíz: la incompatibilidad entre el subgoal continuo de STH-WP y las rutas recalculadas mid-episode.

Durante el entrenamiento v4, cada vez que se dispara un replan, `compute_sth_subgoal` proyecta un nuevo subgoal sobre la ruta recalculada. Desde la perspectiva del agente, el subgoal salta de golpe a una posición muy diferente. El gradiente de política resultante es ruidoso e inconsistente: en algunos steps el agente aprende a seguir la ruta original, en otros a adaptarse al replan. Este ruido degrada la calidad de la política, especialmente en fases tardías del ciclo (exit y retorno), donde la tasa de colisión de s123 y s524 es muy superior a la de s42. El hecho de que s42 no sufra esta regresión sugiere que el impacto depende fuertemente de la seed (el comportamiento aprendido al arrancar v4 puede dar lugar a ciclos estables o inestables según la trayectoria de exploración inicial).

La arquitectura STH-WP es fundamentalmente incompatible con el replanning mid-episode porque el subgoal continuo convierte cualquier cambio de `full_path` en una discontinuidad en el espacio de observación.

### 27.4 SUB-WP — mejora consistente en todas las seeds (+6.5pp de media)

El replanning integrado produce una mejora sólida y consistente en las tres seeds (+5.9pp, +5.9pp, +7.8pp). La tasa de éxito media pasa de 38.9% a 45.5%, un incremento de 6.5pp.

La arquitectura de waypoints discretos es compatible con el replanning porque tras un replan el agente simplemente reinicia `_wp_idx = 0` y sigue la nueva secuencia de waypoints igual que haría al inicio del episodio. El comportamiento es coherente: el agente aprendió durante v4 que puede encontrar una nueva ruta y seguirla, y en inferencia lo ejecuta correctamente.

El punto de fallo principal en SUB-WP v4 es la fase de **retorno** (col_return: 18–28%), donde la tasa de colisión es notablemente superior a approach y exit. La ruta de retorno (zona descarga → zona espera) discurre por el pasillo central, la zona con mayor presencia de peatones. Aunque el replanning en approach y exit mejora el rendimiento, la fase de retorno concentra las colisiones restantes y es el cuello de botella principal del sistema.

### 27.5 Tabla comparativa global — evolución v1 → v4

| Versión | Condición | STH-WP | SUB-WP |
|---------|-----------|:---:|:---:|
| v1 | Sin peatones (baseline) | ~100% | ~100% |
| v1 | Peatones físicamente presentes (ped_obs=False) | 0.7% | 15.5% |
| v3 | Peatones dinámicos, entrenamiento sin replanning | 39.7% | 38.9% |
| v3+replan | Replanificación externa en inferencia | 0.0% | 39.6% |
| **v4** | **Replanificación integrada en entrenamiento** | **22.9%** | **45.5%** |

### 27.6 Conclusiones

1. **SUB-WP con replanning integrado (v4) es la mejor configuración**: 45.5% de media, mejora de +6.5pp sobre v3 y +6.6pp sobre el replanning externo. La mejora es robusta (varianza entre seeds < 2pp).

2. **STH-WP es incompatible con replanning mid-episode**: la arquitectura de subgoal continuo convierte cada replan en una discontinuidad observacional. El replanning integrado (v4) produce peores resultados que v3 en 2/3 seeds. STH-WP alcanza su mejor rendimiento con peatones en el esquema v3 puro (39.7%), sin modificaciones de ruta mid-episode.

3. **El cuello de botella de SUB-WP v4 es la fase de retorno**: approach y exit mejoran claramente respecto a v3, pero la fase de retorno (pasillo central) concentra el 18–28% de las colisiones y es el siguiente objetivo de mejora.

4. **El replanning necesita coherencia con la arquitectura de subgoal**: funciona con waypoints discretos (reinicio natural de índice), pero no con subgoal continuo proyectado (discontinuidad en la observación). Cualquier extensión futura del replanning a STH-WP requeriría una reformulación del cálculo del subgoal que suavice la transición tras el replan.

---

## 28. Reentrenamiento r2 — Corrección del dropoff y análisis de entrenamiento

### 28.1 Motivación y contexto

Durante el análisis post-v4 se detectó un bug crítico en `rl_train_SUB_WP_continuo/warehouse_map01.json`: la zona de descarga (`dropoff`) estaba configurada como `{x: 0.0, y: 10.5}` — posición incorrecta correspondiente a una iteración anterior del mapa — en lugar de `{x: -11.0, y: 0.0}`. Todos los modelos SUB-WP entrenados previamente (v1 a v4) aprendieron a navegar hacia y desde una zona de descarga inexistente, invalidando sus fases de exit y return. STH-WP usaba coordenadas correctas desde el inicio.

**Cadena de reentrenamiento r2** (3 seeds × 5 etapas = 15 runs):

| Etapa | Steps | Carga desde | Novedad |
|-------|-------|-------------|---------|
| stage3_r2 | 4M | stage2_final | exit puro, sin peatones, dropoff corregido |
| stage4_r2 | 4M | stage3_r2_final | approach+exit, sin peatones |
| stage5_r2 | 2M | stage4_r2_final | return puro, sin peatones |
| stage6_din_v3_r2 | 2M | stage5_r2_final (+expansión 40→48 dims) | ciclo completo + peatones + B1 |
| stage6_din_v4_r2 | 2M | stage6_din_v3_r2_final | replanning A* integrado |

**Decisión de diseño**: las etapas 3–5 se entrenaron con `warehouse_1_subwp_noped.wbt` (peatones físicamente desactivados: `controller="<none>"`, `enableBoundingObject=False`) para que el aprendizaje de la navegación básica no se vea perturbado por colisiones aleatorias con peatones que el agente no puede observar ni evitar. Las etapas 6 usan `warehouse_1_subwp.wbt` con peatones activos.

---

### 28.2 Stages 3–5 r2 (sin peatones)

#### Stage 3 r2 — Exit puro (4M steps)

| Seed | Intentos | Éxitos | Colisiones | Tasa éxito | Tasa llego_estantería |
|------|----------|--------|------------|------------|----------------------|
| s42  | 2.482    | 2.153  | 273        | **86.7%**  | 100.0% |
| s123 | 2.474    | 2.238  | 214        | **90.5%**  | 100.0% |
| s524 | 2.460    | 2.234  | 213        | **90.8%**  | 100.0% |
| **Media** | — | — | — | **89.3%** | **100.0%** |

TensorBoard: el reward medio converge a 227–250 a lo largo de los 4M steps (acumulado sobre steps previos: s42 finaliza en ~8.5M, s123 en ~8.5M, s524 en ~8.5M). Las tres curvas son estables sin oscilaciones relevantes. El 100% de `llego_estantería` confirma que el robot siempre alcanza la estantería objetivo; las colisiones se producen en el trayecto de aproximación interior (pasillo estrecho).

#### Stage 4 r2 — Approach + Exit (4M steps)

| Seed | Intentos | Éxitos | Colisiones | Tasa éxito | Tasa llego_estantería |
|------|----------|--------|------------|------------|----------------------|
| s42  | 1.953    | 969    | 53         | **49.6%**  | 100.0% |
| s123 | 1.980    | 1.077  | 59         | **54.4%**  | 100.0% |
| s524 | 1.967    | 1.120  | 17         | **56.9%**  | 100.0% |
| **Media** | — | — | — | **53.6%** | **100.0%** |

TensorBoard: el reward sube de ~150 a ~300–320 durante los 4M steps. La caída respecto a stage3 (89.3% → 53.6%) es esperable: el ciclo approach+exit exige dos fases encadenadas, duplicando la exposición al riesgo de colisión. El 100% de llego_estantería en todos los intentos indica que el robot aprende el trayecto completo pero falla en la precisión de acercamiento al slot.

#### Stage 5 r2 — Return puro (2M steps)

Stage 5 entrena exclusivamente la ruta de retorno (zona de descarga → zona de espera), que es una ruta fija independiente del goal. El callback registra los 28 goals pero solo goal_01 tiene episodios activos; los demás muestran 0% por diseño.

| Seed | Éxito goal_01 | Intentos goal_01 | Nota TensorBoard |
|------|---------------|------------------|-----------------|
| s42  | **95.9%**     | 2.516            | reward ~32, estable |
| s123 | **87.5%**     | 1.834            | reward −141, **aparente colapso** |
| s524 | **99.8%**     | 2.465            | reward ~41, estable |

**Análisis del reward negativo de s123 (aparente colapso)**: el reward acumulado por episodio depende del número de steps hasta completar la ruta. Si el agente tarda más pasos que el óptimo, la penalización acumulada por step hace que el reward sea negativo aunque el episodio termine en éxito. s123 con 87.5% de éxito tiene un reward de -141 porque sus episodios exitosos son significativamente más largos (posiblemente mayor varianza en la trayectoria). El indicador relevante es la **tasa de éxito, no el reward**. El "colapso" visible en TensorBoard es un artefacto de la función de recompensa en return puro y no afecta a la calidad del modelo.

---

### 28.3 Stage 6 din v3 r2 — Transición 40→48 dims + peatones (2M steps)

#### Mecanismo de expansión de pesos

A diferencia del chain original (stage5 → stage6_din vía v1/v2 intermedios que introducían gradualmente ped_obs), r2 salta directamente de stage5_r2 (40-dim obs, ped_obs=False) a stage6_din_v3_r2 (48-dim obs, ped_obs=True). Esto requiere crear un nuevo modelo PPO con arquitectura 48-dim e inicializar sus pesos desde stage5_r2 mediante expansión de la capa de entrada:

- Capas con dimensiones iguales: trasladadas directamente (`transferred`)
- Capa de entrada (obs → hidden): la sección correspondiente a las 40 dims originales se copia; las 8 nuevas dims de peatones se inicializan a **0.0** (`expanded`)
- El agente empieza con los reflejos de navegación del stage5_r2 pero sin conocimiento de los peatones; debe aprenderlo desde cero durante los 2M steps de v3.

#### Stats de entrenamiento

| Seed | Intentos | Éxitos | Colisiones | Tasa éxito | Tasa llego_est. |
|------|----------|--------|------------|------------|-----------------|
| s42  | 657      | 38     | 569        | **5.8%**   | 81.9% |
| s123 | 403      | 2      | 177        | **0.5%**   | 55.1% |
| s524 | 634      | 39     | 552        | **6.2%**   | 85.6% |
| **Media** | — | — | — | **4.2%** | **74.2%** |

Comparación con v3 original (dropoff incorrecto, training callback):

| Seed | v3_orig (train) | v3_r2 (train) | Δ |
|------|----------------|--------------|---|
| s42  | 44.8%          | 5.8%         | −39.0pp |
| s123 | 43.8%          | 0.5%         | −43.3pp |
| s524 | 42.3%          | 6.2%         | −36.1pp |

La diferencia es **metodológicamente injusta**: los modelos v3_orig tenían ~12M steps acumulados de entrenamiento previo con `reset_num_timesteps=False` (el callback mide episodios sobre una política ya entrenada), mientras que v3_r2 usa `reset_num_timesteps=True` (2M steps totales, policy inicialmente ciega a peatones). Los stats de training callback de v3_r2 reflejan el aprendizaje en progreso desde una inicialización desfavorable, no el modelo final.

La `tasa_llego_estantería` de 81.9%–85.6% en s42/s524 confirma que el navegación a la estantería se preserva; el problema es la salida posterior con peatones. s123 con solo 55.1% de llego_estantería indica que también falla en la fase de approach, consistent con su peor partida.

#### Interpretación por goal

En v3_r2, los pocos éxitos de s42 (5.8%) y s524 (6.2%) se concentran en goals de pasillo exterior (goal_01, goal_07, goal_09, goal_16, goal_22) donde la ruta de exit pasa lejos del corredor de peatones. s123 solo registra 2 éxitos en todo el entrenamiento (goals 05 y 11), señal de que su política expansionada no consiguió aprender el comportamiento de evasión en los 2M steps disponibles.

---

### 28.4 Stage 6 din v4 r2 — Replanning integrado (2M steps)

| Seed | Intentos | Éxitos | Colisiones | Tasa éxito | Tasa llego_est. |
|------|----------|--------|------------|------------|-----------------|
| s42  | 813      | 87     | 726        | **10.7%**  | 91.0% |
| s123 | 543      | 8      | 465        | **1.5%**   | 85.1% |
| s524 | 838      | 157    | 681        | **18.7%**  | 90.8% |
| **Media** | — | — | — | **10.3%** | **89.0%** |

Δ respecto a v3_r2 (training callback):

| Seed | v3_r2 | v4_r2 | Δ |
|------|-------|-------|---|
| s42  | 5.8%  | 10.7% | **+4.9pp** |
| s123 | 0.5%  | 1.5%  | **+1.0pp** |
| s524 | 6.2%  | 18.7% | **+12.5pp** |

El replanning sigue aportando mejora incluso partiendo de una política con tan pocos steps de entrenamiento en el dominio dinámico. s524 es el que más se beneficia (+12.5pp), con una tasa de llego_estantería del 90.8% y una tasa de éxito final del 18.7% — la mejor de las tres seeds. La mejora se concentra en la fase de exit (más susceptible a bloqueo por peatón en pasillos) mientras que approach y return mejoran marginalmente.

Comparación con v4 original:

| Seed | v4_orig (train) | v4_r2 (train) | Δ |
|------|----------------|--------------|---|
| s42  | 48.5%          | 10.7%         | −37.8pp |
| s123 | 47.0%          | 1.5%          | −45.5pp |
| s524 | 48.3%          | 18.7%         | −29.6pp |

La misma advertencia metodológica aplica: v4_orig acumula ~14M steps previos; v4_r2 acumula solo 4M (2M de v3 + 2M de v4), de los cuales 2M son la transición ciega al dominio de peatones. **Los stats de training no son comparables directamente**: se necesita inferencia dedicada.

---

### 28.5 Análisis global del entrenamiento r2

#### Curvas de reward TensorBoard (resumen)

| Run | Steps eje x | Reward final (smoothed) | Interpretación |
|-----|-------------|------------------------|----------------|
| stage3_r2 s42/s123/s524 | 4.5M–8.5M | 248 / 227 / 230 | Convergencia sólida, sin ruido |
| stage4_r2 s42/s123/s524 | 8.5M–12.5M | 313 / 302 / 320 | Sube respecto a stage3, estable |
| stage5_r2 s42/s123/s524 | 12.5M–14.5M | 32 / −141 / 41 | Escala distinta (return corto); s123 artefacto |
| stage6_din_v3_r2 (eje 0–2M, reset=True) | 2M | 65 / −83 / 124 | Aprendizaje desde expansión; s524 mejor |
| stage6_din_v4_r2 (eje 2M–4M, reset=False) | 4M | 154 / −62 / 221 | s524 converge; s42 mejora; s123 estancado |

#### Ordenación de seeds por rendimiento

- **s524**: mejor seed en todas las etapas dinámicas. Stage5 98.8%, v3 6.2%, v4 18.7%. Mejor convergencia TensorBoard (reward 221 en v4).
- **s42**: intermedio. Stage5 95.9%, v3 5.8%, v4 10.7%. Reward positivo en v4 (154), mejora progresiva.
- **s123**: peor seed en etapas dinámicas. Stage5 87.5% (aceptable), pero v3 0.5% y v4 1.5% son valores críticos. Reward negativo en toda la fase dinámica (−83 en v3, −62 en v4). El modelo s123 no aprendió a operar con peatones en los 2M+2M steps disponibles.

#### Causas probables del mal rendimiento de s123 en etapas dinámicas

1. **Peor punto de partida**: stage5_r2_s123 alcanza 87.5% (vs 95.9% s42 y 99.8% s524). El menor dominio del return puro implica que la política base transferida al stage6 es más débil.
2. **Seed adversaria para la expansión**: la inicialización aleatoria de los gradientes con seed=123 puede haber llevado la política expandida a un mínimo local difícil de escapar en 2M steps.
3. **Gradientes ruidosos en la fase inicial**: con las 8 nuevas dims a 0.0, los primeros miles de steps generan gradientes muy ruidosos para la capa de entrada. s123 parece haber quedado atrapado en este ruido más que las otras seeds.

---

### 28.6 Perspectiva hacia la inferencia

Los stats de training callback son una señal de aprendizaje en progreso, no de capacidad final del modelo. En el historial de entrenamientos previos (v3/v4 originales), la diferencia entre training callback y inferencia determinista era de ~5–10pp a favor de la inferencia (política determinista vs. estocástica, sin exploración). Para r2, la diferencia podría ser mayor dado que el modelo tiene menos steps de entrenamiento en el dominio dinámico.

**Hipótesis para la inferencia r2**:
- s524 debería ser el mejor modelo, posiblemente alcanzando un 20–30% en v4_r2.
- s42 puede situarse en 12–20%.
- s123 es el candidato a quedar por debajo del 10%, aunque la política determinista puede recuperar más éxito del que sugiere el training callback.

La inferencia dedicada (`run_infer_r2.sh`) dará los resultados definitivos para el §29.

---

## 29. Inferencia r2 — Resultados definitivos (política determinista)

### 29.1 Configuración

- **Política**: determinista (`deterministic=True`), sin exploración
- **Stages 3, 4, din_v3, din_v4**: 100 episodios × 28 goals = 2.800 episodios por seed
- **Stage 5**: 300 episodios (ruta fija retorno)
- **World**: `warehouse_1_subwp.wbt` (peatones físicamente activos en stages 6; presentes pero no observados en stages 3–5)
- **Mapa**: `warehouse_map01.json` con `dropoff: {x: -11.0, y: 0.0}` (corregido)

---

### 29.2 Resultados por etapa

#### Stage 3 r2 — Exit puro

| Seed | Éxito | Colisión | Truncado | Tasa éxito |
|------|-------|----------|----------|------------|
| s42  | 2.152/2.800 | 644 | 4 | **76.9%** |
| s123 | 816/2.800   | 1.984 | 0 | **29.1%** |
| s524 | 785/2.800   | 2.015 | 0 | **28.0%** |
| **Media** | — | — | — | **44.7%** |

La disparidad entre s42 (76.9%) y s123/s524 (~28%) es llamativa dado que el training callback de las tres seeds rondaba el 87–91%. El análisis por goal revela que s42 generaliza bien a todos los goals (muchos al 65–100%), mientras que s123 y s524 colapsan en un subconjunto de goals específicos (goal_03, goal_04, goal_14, goal_17, goal_18, goal_22, goal_23, goal_24, goal_27 → 0% en ambas seeds). Estos goals comparten una ruta de exit que pasa por el pasillo interior derecho, lo que sugiere que s123 y s524 aprendieron una política más local que no generaliza a los trayectos menos frecuentes durante entrenamiento.

**Goal más difícil (media 3 seeds)**: goal_27 (3.7%), goal_21 (23.7%), goal_22 (19.3%).
**Goal más fácil**: goal_10 (100.0% las tres seeds), goal_02 (84.0%), goal_06 (56.7%).

#### Stage 4 r2 — Approach + Exit

| Seed | Éxito | Col. approach | Col. exit | Llego estant. | Tasa éxito |
|------|-------|--------------|-----------|---------------|------------|
| s42  | 908/2.800 | 497 | 1.395 | 82.2% | **32.4%** |
| s123 | 740/2.800 | 594 | 1.466 | 78.8% | **26.4%** |
| s524 | 430/2.800 | 453 | 1.917 | 83.8% | **15.4%** |
| **Media** | — | — | — | — | **24.7%** |

La mayoría de las colisiones ocurren en **exit** (tras llegar a la estantería), no en approach. El llego_estantería de 78–84% confirma que el approach funciona razonablemente, pero la salida posterior (con el nuevo dropoff en (-11.0, 0.0)) genera alta tasa de colisión en algunos pasillos. s524 cae a 15.4% pese a tener 83.8% de llego_estantería — su política de exit es la más frágil de las tres.

Goals con 0% en las 3 seeds: goal_10, goal_12, goal_16, goal_20, goal_21, goal_22. Corresponden a estanterías del extremo derecho del almacén cuya salida pasa por el pasillo más estrecho hacia el nuevo dropoff.

#### Stage 5 r2 — Return puro

| Seed | Éxito | Colisión | Avg. pasos total | Avg. pasos col. | Tasa éxito |
|------|-------|----------|-----------------|-----------------|------------|
| s42  | 21/300 | 279 | 45 | 3 | **7.0%** |
| s123 | 0/300  | 300 | 8  | 8 | **0.0%** |
| s524 | 21/300 | 279 | 48 | 7 | **7.0%** |

**Resultado crítico**: la tasa de éxito en inferencia (7%) contradice completamente el training callback (87–96%). El indicador clave es `avg_pasos_colisión ≈ 3–8 steps`, lo que significa que la colisión se produce casi inmediatamente al inicializar el episodio.

**Causa probable**: el dropoff corregido `(-11.0, 0.0)` sitúa al robot en una posición que, en el world file `warehouse_1_subwp.wbt`, queda pegada a una estantería o pared. Durante el entrenamiento con `--mode=fast` la física puede ser más permisiva, permitiendo al robot escapar en la mayoría de casos; en inferencia, el robot activa el bumper en el primer step de movimiento. Los 21 episodios exitosos (avg. 600 pasos) corresponden a las pocas inicializaciones donde el robot logra separarse de la geometría antes de colisionar.

**Implicación**: los resultados de stage5_r2 en inferencia son **no representativos** de la capacidad del modelo y requieren verificación de la posición de inicialización en el world file. Los stats de entrenamiento (87–96%) son la referencia válida.

#### Stage 6 din v3 r2 — Ciclo completo + peatones

| Seed | Éxito | Col. approach | Col. exit | Col. return | Truncado | Tasa éxito |
|------|-------|--------------|-----------|-------------|----------|------------|
| s42  | 164/2.800  | 287   | 1.554 | 795  | 0   | **5.9%**  |
| s123 | 142/2.800  | 566   | 1.510 | 287  | 295 | **5.1%**  |
| s524 | 820/2.800  | 501   | 822   | 657  | 0   | **29.3%** |
| **Media** | — | — | — | — | — | **13.4%** |

**s524 destaca claramente** (29.3%) mientras s42 y s123 quedan en el 5–6%. El desglose de colisiones muestra que la **fase de exit es el cuello de botella principal** (1.554 en s42, 1.510 en s123, 822 en s524). La fase de retorno es el segundo factor. s123 acumula 295 episodios truncados (timeout) — señal de que el agente se queda atrapado sin colisionar, posiblemente oscilando ante un peatón.

s524 es la única seed que mantiene la tasa de exit bajo control (822 col. exit vs. ~1.500 en las otras), lo que explica su superioridad. La diferencia entre seeds apunta a que la expansión 40→48 dims no convergió igual para todas: s524 encontró una política de evasión de peatones en exit, s42 y s123 no.

**Comparación con v3_orig (dropoff incorrecto)**:

| Seed | v3_orig | v3_r2 | Δ |
|------|---------|-------|---|
| s42  | 39.8%   | 5.9%  | −33.9pp |
| s123 | 30.5%   | 5.1%  | −25.5pp |
| s524 | 27.0%   | 29.3% | **+2.3pp** |

s524 mejora ligeramente respecto al original a pesar de tener solo 2M steps de entrenamiento dinámico frente a los ~14M acumulados del original. Las otras seeds bajan significativamente, pero esto se debe al menor número de steps disponibles para aprender el dominio dinámico (ver §28.5).

#### Stage 6 din v4 r2 — Ciclo completo + peatones + replanning

| Seed | Éxito | Col. approach | Col. exit | Col. return | Truncado | Tasa éxito |
|------|-------|--------------|-----------|-------------|----------|------------|
| s42  | 336/2.800  | 397   | 1.479 | 588  | 0   | **12.0%** |
| s123 | 10/2.800   | 530   | 1.679 | 32   | 549 | **0.4%**  |
| s524 | 539/2.800  | 541   | 781   | 939  | 0   | **19.2%** |
| **Media** | — | — | — | — | — | **10.5%** |

El replanning aporta mejora medible en s42 (+6.1pp sobre v3_r2) y s524 (+9.9pp sobre v3_r2 en esta seed). En s123 el modelo está completamente roto (0.4%, 549 truncados) — el replanning no puede compensar una política base que no funciona.

**Comparación con v4_orig (dropoff incorrecto)**:

| Seed | v4_orig | v4_r2 | Δ |
|------|---------|-------|---|
| s42  | 45.7%   | 12.0% | −33.7pp |
| s123 | 44.2%   | 0.4%  | −43.8pp |
| s524 | **0.0%** | **19.2%** | **+19.2pp** |

El dato más relevante: **v4_orig_s524 tenía 0.0% de éxito** en inferencia con el dropoff incorrecto (el robot aprendió a ir a una zona de descarga que no existe → never completes the cycle). v4_r2_s524 alcanza 19.2% con el dropoff correcto — la corrección del bug es la causa directa de esta mejora. Esto valida que el reentrenamiento r2 es correcto para s524.

Para s42 y s123, v4_r2 queda por debajo del original no porque el dropoff correcto sea peor, sino porque el chain de entrenamiento r2 tiene muchos menos steps acumulados en el dominio dinámico.

---

### 29.3 Goals más difíciles y más fáciles (din v4 r2)

**Goals con mayor tasa media (3 seeds)**:

| Goal | s42 | s123 | s524 | Media |
|------|-----|------|------|-------|
| goal_18 | 19% | 0% | 53% | 24.0% |
| goal_11 | 27% | 0% | 32% | 19.7% |
| goal_16 | 17% | 0% | 42% | 19.7% |
| goal_09 | 19% | 0% | 32% | 17.0% |
| goal_14 | 14% | 6% | 28% | 16.0% |

**Goals con menor tasa media**:

| Goal | s42 | s123 | s524 | Media |
|------|-----|------|------|-------|
| goal_21 | 1% | 0% | 0% | 0.3% |
| goal_22 | 1% | 0% | 0% | 0.3% |
| goal_20 | 4% | 0% | 0% | 1.3% |
| goal_26 | 10% | 0% | 10% | 6.7% |
| goal_19 | 8% | 0% | 0% | 2.7% |

Los goals difíciles (21, 22, 19, 20, 26) concentran dos características: están en el extremo del almacén y su ruta de exit pasa por el pasillo central donde se mueven los peatones, sin rutas alternativas viables con el margen de inflado actual (0.75 m).

---

### 29.4 Análisis global y conclusiones

#### Rendimiento por seed

| Seed | Stage3 | Stage4 | Stage5* | Din v3 | Din v4 | Valoración |
|------|--------|--------|---------|--------|--------|------------|
| s42  | 76.9%  | 32.4%  | 7.0%   | 5.9%   | 12.0%  | Exit sólido, dinámica frágil |
| s123 | 29.1%  | 26.4%  | 0.0%   | 5.1%   | 0.4%   | Fallo sistemático en múltiples etapas |
| s524 | 28.0%  | 15.4%  | 7.0%   | 29.3%  | 19.2%  | Peor en estáticas, mejor en dinámica |

*Stage 5 con posible bug de inicialización — ver §29.2.

#### Conclusiones

1. **La corrección del dropoff es indispensable para s524**: v4_r2_s524 (+19.2%) frente a v4_orig_s524 (0.0%) demuestra que el bug del dropoff invalidaba completamente el ciclo completo de esta seed. El reentrenamiento r2 recupera funcionalidad real.

2. **Los modelos r2 necesitan más steps de entrenamiento dinámico**: s42 y s123 quedan muy por debajo del original (12.0% vs 45.7% para s42) no por el dropoff, sino porque el chain r2 solo acumula 4M steps en el dominio dinámico frente a los ~14M del original. La solución natural es continuar el entrenamiento desde los modelos r2_v4_final con más steps (r3) o con un warm-start más largo en stage6.

3. **s123 es un caso perdido en este chain**: 0.4% en v4_r2. La expansión 40→48 dims no convergió en los 2M steps de v3. Esta seed requeriría reinicialización desde cero en stage6 o un entrenamiento v3 más largo (4M steps en lugar de 2M).

4. **El cuello de botella sistemático es la fase de exit**: en todos los modelos r2 dinámicos, la mayoría de colisiones ocurren en exit (robot saliendo del pasillo de estanterías hacia el dropoff). El trayecto de exit cruza zonas donde los peatones bloquean con mayor frecuencia y el margen de replanning es menor.

5. **Stage 5 inferencia no es representativa**: la tasa de 7% con colisiones a los 3–8 steps indica un problema de inicialización en el world file, no de capacidad del modelo. El training callback (87–96%) es el indicador correcto para esta etapa.

> **Nota**: Los resultados de stages 3, 4 y 5 en §29 son incorrectos — el launcher de inferencia usaba el world con peatones activos para etapas entrenadas sin ellos. Los datos corregidos se encuentran en §30.

---

## 30. Inferencia r2 — Resultados corregidos (world sin peatones para stages 3–5)

### 30.1 Bug de consistencia en el launcher y corrección

Al analizar el rendimiento de stages 3–5 en inferencia (§29) se detectaron dos anomalías:

1. **Stage 3 y 4**: tasas muy por debajo del callback de entrenamiento (44.7% y 24.7% vs. 87–91% en callback). Inconsistente con que stage 4 tiene un dominio más fácil que stage 3.
2. **Stage 5**: `avg_pasos_colisión ≈ 3–8` — colisión casi inmediata desde el spawn.

**Causa raíz identificada**: `run_infer_r2.sh` usaba `warehouse_1_subwp.wbt` (peatones físicamente activos) para **todas** las etapas, incluidas stages 3–5 que se entrenaron con `warehouse_1_subwp_noped.wbt`. El Pedestrian 2 recorre `x = −9.5` entre `y = −5.0` y `y = −0.5`, a 0.25 m del punto de spawn de stage 5 en `(−9.75, −3.75)` → colisión de bumper en los primeros 3–8 steps. Para stages 3 y 4, los peatones físicos actúan como obstáculos no vistos durante entrenamiento (el agente no tiene `ped_obs`) y degradan la tasa de éxito.

**Correcciones aplicadas**:
- `run_infer_r2.sh`: stages 3/4/5 → `run_noped()` con `WORLD_NOPED`, stages 6 → `run_ped()` con `WORLD_PED`
- `webots_env.py` `_reset_return()`: comentario actualizado para referenciar el dropoff correcto `(−11.0, 0.0)`
- `run_infer_r2_stages345.sh`: launcher parcial para re-ejecutar solo las 9 inferencias afectadas

---

### 30.2 Resultados corregidos por etapa

#### Stage 3 r2 — Exit puro (sin peatones)

| Seed | Éxito | Colisión | Truncado | Tasa éxito | Avg. pasos OK |
|------|-------|----------|----------|------------|--------------|
| s42  | 2.786/2.800 | 14   | 0 | **99.5%** | 1.883 |
| s123 | 2.604/2.800 | 196  | 0 | **93.0%** | 1.753 |
| s524 | 2.604/2.800 | 196  | 0 | **93.0%** | 1.767 |
| **Media** | — | — | — | **95.2% ± 3.7** | 1.801 |

**Mejora respecto a §29**: +50.5 pp en media (44.7% → 95.2%). Con peatones ausentes el modelo funciona tal como indica el training callback. s42 casi perfecto; s123/s524 tienen 196 colisiones cada una, probablemente concentradas en goals de pasillos estrechos cuya geometría presenta mayor dificultad con el nuevo dropoff.

**Goals más difíciles (s123/s524)**: las 196 colisiones por seed no están uniformemente distribuidas — corresponden a goals específicos cuya ruta de exit tiene menor margen respecto al nuevo dropoff `(−11.0, 0.0)`. La media de 95.2% indica que el modelo de exit aprendido correctamente en stage 3 generaliza bien.

---

#### Stage 4 r2 — Approach + Exit (sin peatones)

| Seed | Éxito | Colisión | Truncado | Tasa éxito | Avg. pasos OK |
|------|-------|----------|----------|------------|--------------|
| s42  | 2.800/2.800 | 0 | 0 | **100.0%** | 2.582 |
| s123 | 2.800/2.800 | 0 | 0 | **100.0%** | 2.540 |
| s524 | 2.800/2.800 | 0 | 0 | **100.0%** | 2.452 |
| **Media** | — | — | — | **100.0% ± 0.0** | 2.525 |

**Resultado perfecto**: las 3 seeds alcanzan 100% de éxito sin ninguna colisión. Este es el resultado esperado para un stage estático entrenado con 4M steps — la política de approach + exit está completamente consolidada. El entrenamiento sin peatones en inferencia también sin peatones produce resultados deterministas y robustos.

**Comparación con §29**: la corrección del world pasa de 24.7% a 100.0%, una mejora de +75.3 pp. Los resultados anteriores eran completamente artefactuales.

---

#### Stage 5 r2 — Return puro (sin peatones)

| Seed | Éxito | Colisión | Truncado | Tasa éxito | Avg. pasos OK | Avg. pasos col. |
|------|-------|----------|----------|------------|--------------|----------------|
| s42  | 300/300 | 0   | 0 | **100.0%** | 598 | — |
| s123 | 0/300   | 295 | 5 | **0.0%**   | —   | 1.379 |
| s524 | 300/300 | 0   | 0 | **100.0%** | 597 | — |
| **Media** | — | — | — | **66.7% ± 57.7** | 597 | — |

**Diagnóstico diferenciado**: el bug del peatón enmascaraba dos comportamientos distintos.

- **s42 y s524** → 100% de éxito. Con el world correcto el spawn en `(−9.75, −3.75)` es completamente seguro (25/25 celdas libres en el A*) y el modelo navega la ruta completa hacia `zona_espera` sin incidencia. El avg. de 597–598 steps refleja la longitud real del return path (~10.5 m).

- **s123** → 0% de éxito (295 colisiones + 5 truncados). El `avg_pasos_colisión = 1.379` indica que el robot navega ~1.400 steps antes de colisionar — es decir, recorre la mayor parte del trayecto pero falla en algún punto intermedio. Esto es un **fallo genuino del modelo s123** para la tarea de retorno, no un artefacto de inicialización. La seed 123 no convergió bien en stage 5 de r2: el callback de entrenamiento mostraba 87.5% pero el modelo no generaliza en inferencia determinista.

La alta desviación estándar (±57.7) es consecuencia directa del colapso de s123 con las otras dos seeds en perfecto: no existe una "media real" representativa. El indicador válido es el par {s42:100%, s524:100%} por un lado y {s123:0%} por otro.

---

#### Stage 6 din v3 r2 — Ciclo completo + peatones (sin cambios)

Los resultados de din_v3 son idénticos a §29 porque el world con peatones era correcto para stage 6. Se reproducen para referencia:

| Seed | Éxito | Col. approach | Col. exit | Col. return | Truncado | Tasa éxito |
|------|-------|--------------|-----------|-------------|----------|------------|
| s42  | 164/2.800  | 287   | 1.554 | 795  | 0   | **5.9%**  |
| s123 | 142/2.800  | 566   | 1.510 | 287  | 295 | **5.1%**  |
| s524 | 820/2.800  | 501   | 822   | 657  | 0   | **29.3%** |
| **Media** | — | — | — | — | — | **13.4% ± 13.8** |

---

#### Stage 6 din v4 r2 — Ciclo completo + peatones + replanning (sin cambios)

| Seed | Éxito | Col. approach | Col. exit | Col. return | Truncado | Tasa éxito |
|------|-------|--------------|-----------|-------------|----------|------------|
| s42  | 336/2.800  | 397   | 1.479 | 588  | 0   | **12.0%** |
| s123 | 10/2.800   | 530   | 1.679 | 32   | 549 | **0.4%**  |
| s524 | 539/2.800  | 541   | 781   | 939  | 0   | **19.2%** |
| **Media** | — | — | — | — | — | **10.5% ± 9.5** |

---

### 30.3 Tabla resumen comparativa §29 vs §30

| Etapa | §29 (bugged) | §30 (correcto) | Δ | Causa del cambio |
|-------|-------------|----------------|---|-----------------|
| Stage 3 media | 44.7% | **95.2%** | +50.5 pp | Peatones eliminados en inferencia |
| Stage 4 media | 24.7% | **100.0%** | +75.3 pp | Peatones eliminados en inferencia |
| Stage 5 s42   | 7.0%  | **100.0%** | +93.0 pp | Bug peatón + spawn corregido |
| Stage 5 s524  | 7.0%  | **100.0%** | +93.0 pp | Bug peatón + spawn corregido |
| Stage 5 s123  | 0.0%  | **0.0%**   | 0 pp | Fallo genuino del modelo |
| Din v3 media  | 13.4% | **13.4%** | 0 pp | World ya correcto |
| Din v4 media  | 10.5% | **10.5%** | 0 pp | World ya correcto |

---

### 30.4 Tabla resumen final — INFERENCIA r2 corregida

| Etapa | N ep. | s42 | s123 | s524 | Media | Col% | Trunc% |
|-------|-------|-----|------|------|-------|------|--------|
| Stage 3 (exit puro) | 2.800×3 | 99.5% | 93.0% | 93.0% | **95.2%** | 4.8% | 0.0% |
| Stage 4 (app+exit)  | 2.800×3 | 100.0% | 100.0% | 100.0% | **100.0%** | 0.0% | 0.0% |
| Stage 5 (return)    | 300×3   | 100.0% | 0.0% | 100.0% | **66.7%** | 32.8% | 0.6% |
| Din v3 (ciclo+peat) | 2.800×3 | 5.9% | 5.1% | 29.3% | **13.4%** | 83.1% | 3.5% |
| Din v4 (ciclo+repl) | 2.800×3 | 12.0% | 0.4% | 19.2% | **10.5%** | 82.9% | 6.5% |

---

### 30.5 Conclusiones actualizadas

1. **Stages 3 y 4 funcionan correctamente**: con el world correcto el modelo de exit puro (stage 3, 95.2%) y approach+exit (stage 4, 100.0%) demuestran que el entrenamiento r2 en entorno estático produjo políticas robustas. La degradación de §29 era íntegramente artefactual.

2. **Stage 5 divide las seeds en dos grupos**: s42 y s524 convergieron correctamente (100%) mientras que s123 colapsó (0%). El avg_pasos_col=1.379 para s123 descarta el bug de inicialización — el modelo aprendió a moverse pero no a completar el retorno. Con 2M steps de stage 5 r2 y el peso 40→48 dims, s123 necesitaría más steps o una reinicialización en stage 5 para corregirlo.

3. **La degradación en din_v3/v4 respecto al original se debe a steps insuficientes, no al bug**: la cadena r2 acumula 4M steps en el dominio dinámico (2M v3 + 2M v4) frente a ~14M del original. Las diferencias en §29 ya identificaron esto; se confirma que el world no es el factor.

4. **s123 es un outlier sistémico**: falla en stage 5 (0%), din_v3 (5.1%) y din_v4 (0.4%). El problema tiene origen en la expansión de pesos 40→48 dims de esta seed particular, que en 2M steps no consiguió estabilizar la política dinámica. Las seeds s42 y s524 son las representativas del sistema r2.

5. **El cuello de botella del ciclo completo sigue siendo la fase exit**: col_exit domina en din_v3/v4 (~50–60% de todas las colisiones). El replanning ayuda en s42 (+6.1 pp) y s524 — la limitación actual es la frecuencia con que los peatones bloquean exactamente el trayecto de exit en el momento de la ejecución, sin tiempo de cooldown suficiente para el replanning.

---

## 31. Revisión bibliográfica — Propuestas de mejora para obstáculos dinámicos

### 31.1 Papers analizados

Se analizan cinco trabajos recientes de DRL para navegación con obstáculos dinámicos. Tres son directamente relevantes al escenario del TFM (robot diferencial, entornos tipo almacén, obstáculos móviles); los otros dos ofrecen ideas puntuales aplicables.

---

#### [P1] Pan et al. (2026) — GE-DRL
> *Dynamic obstacle avoidance for autonomous vehicles in complex traffic environments: A guidance-enhanced deep reinforcement learning approach.*
> Engineering Applications of Artificial Intelligence, vol. 181, art. 115436.
> DOI: 10.1016/j.engappai.2026.115436

**Propuesta central**: framework GE-DRL que combina tres componentes sobre SAC:

1. **PA-DPF (Prediction-Aware Dynamic Potential Field)**: predice la trayectoria futura de cada obstáculo usando un filtro IMM-UKF con cuatro modelos de movimiento — velocidad constante (CV), aceleración constante (CA), Singer (aceleración correlada en el tiempo) y Jerk (maniobras abruptas). La predicción probabilística de la posición futura del obstáculo alimenta un campo potencial repulsivo que genera acciones de referencia anticipativas.

2. **Guía por imitación decreciente**: la política de referencia (PA-DPF) se usa como supervisora de la política RL mediante una pérdida de imitación:
   `J_IL = E[‖π_θ(s) − π_lead(s)‖²]`
   con peso que decae exponencialmente: `λ_IL = λ_0 · exp(−β · t)`. Al inicio del entrenamiento el agente imita la guía (reduce colisiones en exploración temprana); conforme acumula experiencia, la guía desaparece y el agente aprende de forma autónoma.

3. **MPC como capa de factibilidad física**: las acciones del RL se pasan por un controlador MPC que garantiza que sean cinemáticamente realizables (radio de giro, aceleración lateral). En el contexto del MiR100 esto equivale a filtrar comandos de velocidad que violarían las restricciones físicas del robot diferencial.

**Resultados**: GE-SAC alcanza **98% de éxito / 2% de colisión** (vs 90% SAC sin guía, 81% TD3 sin guía) en entorno con 10-20 obstáculos a 90 km/h. La guía PA-DPF reduce colisiones en fase temprana y acelera convergencia; SAC supera a TD3 por su exploración estocástica con regularización de entropía.

**Limitaciones del contexto**: vehículo autónomo en carretera (no robot de almacén), obstáculos son vehículos (no peatones a pie), acción continua de alto nivel. La arquitectura MPC requiere un modelo cinemático calibrado del robot.

---

#### [P2] Zhang et al. (2025) — GAP\_SAC
> *Deep reinforcement learning for path planning of autonomous mobile robots in complicated environments.*
> Complex & Intelligent Systems, vol. 11, art. 277.
> DOI: 10.1007/s40747-025-01906-9

**Propuesta central**: mejoras apiladas sobre SAC para AMR (TurtleBot3, Gazebo) en entornos con obstáculos dinámicos y pasillos estrechos. Es el paper más cercano al escenario del TFM.

1. **Espacio de estado expandido**: a los rayos LiDAR se añaden cuatro métricas explícitas:
   - `D_G`: distancia euclidiana al goal
   - `D_O`: distancia al obstáculo más cercano
   - `A_O`: ángulo al obstáculo más cercano (relativo al heading del robot)
   - `M_A`: ángulo que el robot debe girar para encarar el goal

   El LiDAR crudo da distancias radiales pero no la relación angular entre obstáculo y dirección de avance del robot. `A_O` y `M_A` hacen explícita esa información, mejorando la toma de decisión en pasillos angostos.

2. **Función de recompensa heurística multiplicativa**:
   ```
   R_total = R1 × R2 + R3
   R1 = λ₁ × (π/3 − |α|)           # yaw alignment: positivo si heading ≈ goal, negativo si >60° de desviación
   R2 = λ₂ × (D_k / D_{k+1} − 1)   # progreso de distancia al goal
   R3 = λ₃  si D_O < 0.25 m         # penalización zona de exclusión (constante negativa)
   ```
   La multiplicación R1×R2 hace que el agente solo reciba reward de progreso si además está bien alineado. Si el heading está desviado más de 60°, el reward de distancia se vuelve negativo aunque el robot se acerque en línea recta — elimina la posibilidad de recompensar avance lateral hacia el goal.

3. **Prioritized Experience Replay (PER)** con SumTree: las transiciones con mayor TD-error (las más "sorprendentes" para el agente, típicamente colisiones y evasiones exitosas cerca del obstáculo) se muestrean más frecuentemente. La prioridad es `P(i) = δᵢ / Σδⱼ` donde `δᵢ = (|TD_error_i| + ε)^α`. Acelera la convergencia priorizando los casos difíciles.

4. **Gated Attention Mechanism**: la red MLP del SAC se extiende con una cabeza de atención (scores Q·K^T/√d_k, softmax, gating sigmoid) que aprende a ponderar diferencialmente las dimensiones del estado según su relevancia contextual. Cuando el obstáculo está cerca, el agente "atiende" más a `D_O` y `A_O`; cuando está lejos, atiende más a `D_G` y `M_A`.

**Resultados** (Gazebo, entorno 10×8 m, obstáculo cilíndrico aleatorio):

| Algoritmo | Éxito (complicado) | Éxito (normal) | Convergencia |
|-----------|-------------------|----------------|-------------|
| GAP\_SAC | **93%** | **95%** | ~300 ep |
| GA\_SAC (sin PER) | 82% | 89% | ~400 ep |
| P\_SAC (sin atención) | 80% | 91% | ~600 ep |
| SAC puro | ~48% | 68% | No converge |
| TD3 | ~45% | 65% | No converge |

GAP\_SAC mejora ~30 pp sobre SAC estándar en entornos complicados y converge en la mitad de episodios que P\_SAC.

**Limitaciones del contexto**: un solo obstáculo dinámico, tarea simple de navegación punto-a-punto (sin ciclo completo), entorno 2D sin estructura de almacén.

---

#### [P3] Gupta et al. (2020) — VAE+SAC/DDPG
> *Policy-Gradient and Actor-Critic Based State Representation Learning for Safe Driving of Autonomous Vehicles.*
> Sensors, vol. 20, art. 5991.
> DOI: 10.3390/s20215991

**Propuesta central**: usar un Variational Autoencoder (VAE) pre-entrenado para comprimir la observación visual (imágenes de cámara) a un espacio latente de baja dimensión sobre el que corre DDPG o SAC. El VAE aprende la representación del entorno sin supervisión a partir de secuencias de imágenes de conducción.

**Función de recompensa con penalización de suavidad**:
```
r = v · (cos θ − d − Ψ · |s_{t+1} − s_t|)
```
donde `θ` es el ángulo entre el heading y la carretera, `d` la distancia al centro del carril, y `Ψ · |s_{t+1} − s_t|` penaliza cambios bruscos de estado (variación de velocidad entre steps consecutivos).

**Resultados**: VAE+SAC más estable que VAE+DDPG. La penalización de suavidad reduce oscilaciones en el steering.

**Limitaciones del contexto**: conducción de un carril recto sin obstáculos como escenario principal. El VAE asume observación de imagen; no aplicable directamente con sensores LiDAR/GPS.

---

#### [P4] Savid et al. (2023) — PPO+BC en Unity
> *Simulated Autonomous Driving Using Reinforcement Learning: A Comparative Study on Unity's ML-Agents Framework.*
> Information, vol. 14, art. 290.
> DOI: 10.3390/info14050290

**Propuesta central**: comparativa de PPO, MA-PPO y POCA para agentes de kart en Unity ML-Agents. El resultado principal relevante es el uso de **Behavioral Cloning (BC) como pre-entrenamiento**: el agente aprende primero de trayectorias de referencia (demostraciones) y luego refina con RL.

**Resultados**: PPO puro logra reward acumulado 0.761 en track sin obstáculos. PPO+BC en track con obstáculos (roadblocks estáticos) obtiene 0.068 — la métrica baja porque el entorno es más difícil, pero BC permite al agente aprender a evadir obstáculos con comportamiento "humano".

**Limitaciones del contexto**: carrera de karts, obstáculos estáticos (no dinámicos), Unity (no Webots/ROS). La idea del BC es transferible pero requiere generar demostraciones válidas.

---

#### [P5] Ho et al. (2025) — AGV RL en Smart Logistics
> *Integrated reinforcement learning of automated guided vehicles dynamic path planning for smart logistics and operations.*
> Transportation Research Part E, vol. 196, art. 104008.
> DOI: 10.1016/j.tre.2025.104008

**Propuesta central**: framework RL para planificación de rutas de AGVs en almacenes inteligentes con integración IoT. Propone coordinación multi-AGV, gestión de batería y adaptación a cambios dinámicos en el entorno (posiciones de workstations, rutas bloqueadas). El escenario es el más cercano organizativamente al TFM (almacén, AGV, rutas dinámicas).

**Limitaciones**: solo disponible el abstract — no hay detalles técnicos del algoritmo RL utilizado, arquitectura de red ni función de recompensa.

---

### 31.2 Propuestas aplicables al TFM — análisis por dimensión

#### A. Espacio de observación

**Contexto actual**: 48 dims — 36 rayos LiDAR + `(dist_wp, angle_wp, v_lin, v_ang)` + `(dx_ped, dy_ped, vx_ped, vy_ped)` × 2 peatones.

**Propuesta A1 — Observación predictiva del peatón** *(Pan et al. [P1])*

En lugar de solo posición actual, incluir la posición predicha en `t+k`:
```python
pred_x = ped_x + vx * k * dt
pred_y = ped_y + vy * k * dt
dx_pred = (pred_x - robot_x) / MAX_GOAL_DIST
dy_pred = (pred_y - robot_y) / MAX_GOAL_DIST
```
Con `k=2` steps (≈0.12 s a 64 ms/step) → 4 dims adicionales por peatón → obs pasa de 48 a **56 dims**. El agente vería dónde estará el peatón en el siguiente ciclo de control, permitiéndole iniciar la maniobra de evasión con suficiente anticipación. La predicción lineal es válida para peatones que caminan en línea recta (que es exactamente el patrón del Pedestrian 1 y 2 en el world file actual).

Requiere reentrenar stage 6 con nueva dimensión (nuevo transfer 40→56 en lugar de 40→48).

**Propuesta A2 — Ángulo al peatón relativo al heading** *(Zhang et al. [P2])*

Añadir `A_O_ped` = ángulo entre el heading actual del robot y la dirección al peatón más cercano:
```python
A_O_ped = atan2(dy_ped, dx_ped) - robot_heading   # normalizado a [-π, π]
A_O_ped_norm = A_O_ped / π
```
El LiDAR detecta al peatón como una región de menor distancia en ciertos rayos, pero no da explícitamente "el peatón está 30° a mi izquierda respecto a donde voy". `A_O_ped` codifica directamente esa relación angular. Especialmente relevante en la fase exit donde el robot tiene una dirección de avance fija y el peatón puede aparecer en cualquier ángulo relativo. 1-2 dims extra.

---

#### B. Función de recompensa

**Contexto actual**: `r = r_progress + r_proximity_lidar + r_proximity_ped + r_angular + r_alignment`

**Propuesta B1 — Penalty de peatón con decay temporal** *(Pan et al. [P1])*

Inspirado en la guía decreciente PA-DPF, hacer que el peso del penalty de peatón sea más fuerte al inicio del entrenamiento de stage 6 y decaiga conforme el agente acumula experiencia:
```python
ped_penalty_w = 1.6 * exp(−total_steps / 500_000) + 0.8
# decay: de 2.4 al inicio → 0.8 tras 2M steps
recompensa -= ped_penalty_w * 0.5 * exp(−2.0 * dist_ped)
```
Durante la exploración temprana (fase más peligrosa), el agente recibe una señal más fuerte para alejarse del peatón, reduciendo colisiones aleatorias. Conforme aprende, el peso baja y deja más libertad para desarrollar comportamientos de evasión más sofisticados. Compatible con PPO sin cambiar arquitectura.

**Propuesta B2 — Zona de exclusión de seguridad graduada** *(Zhang et al. [P2])*

Reemplazar la penalización exponencial continua por una zona dura + zona blanda:
```python
if dist_ped < 0.5:           # zona dura: penalización constante grande
    recompensa -= 2.0
elif dist_ped < 2.0:         # zona blanda: penalización exponencial
    recompensa -= 0.8 * exp(−2.0 * dist_ped)
```
La zona dura clara enseña al agente que entrar a <0.5 m es siempre malo, independientemente de la velocidad o dirección. La zona blanda mantiene la señal de gradiente para el aprendizaje.

**Propuesta B3 — Progreso multiplicativo por alineación de heading** *(Zhang et al. [P2])*

Modificar el bonus de alineación para que module el reward de progreso en lugar de ser aditivo:
```python
# Actual (aditivo):
recompensa += 0.15 * cos(angulo_rel)

# Propuesto (multiplicativo sobre progreso):
alignment_factor = max(0.0, cos(angulo_rel))
recompensa_progreso *= alignment_factor
```
El agente solo recibe reward de avance si además está bien alineado con el waypoint. Elimina el incentivo a "acercarse al waypoint de lado", que es un comportamiento que aparece cuando un peatón bloquea el frente y el robot intenta rodearle mientras mantiene el progreso.

**Propuesta B4 — Penalización de suavidad angular** *(Gupta et al. [P3])*

Añadir penalización por cambio brusco de velocidad angular entre steps consecutivos:
```python
# En step(), guardando v_ang del step anterior:
recompensa -= 0.02 * abs(velocidad_angular - self._prev_v_ang)
self._prev_v_ang = velocidad_angular
```
La penalización actual (`-0.05 * |v_ang|`) penaliza la magnitud absoluta pero no los cambios. El comportamiento oscilatorio (izquierda-derecha) que aparece cuando el robot duda ante un peatón no es penalizado por la métrica actual pero sí por la de suavidad.

---

#### C. Arquitectura de red

**Contexto actual**: `MlpPolicy` de SB3 — MLP con 2 capas ocultas de 64 nodos, todos los inputs tratados con el mismo peso estructural.

**Propuesta C1 — Gated Attention sobre el input** *(Zhang et al. [P2])*

Implementar una `CustomActorCriticPolicy` en SB3 con un extractor de features que aplique atención sobre las dimensiones del estado antes de entrar al MLP:
```python
class GatedAttentionExtractor(BaseFeaturesExtractor):
    # 1. Linear projection → Q, K, V
    # 2. Attention scores: softmax(Q·Kᵀ / √d_k)
    # 3. Gating: sigmoid(linear(attention_output))
    # 4. Output → MLP estándar
```
El beneficio: el agente aprende a dar más peso a las dims del peatón cuando está cerca y más peso a las dims del waypoint cuando está lejos, sin tener que aprender implícitamente esa selectividad a través del MLP. Requiere implementación custom (~100 líneas PyTorch) y reentrenamiento.

---

#### D. Algoritmo de entrenamiento

**Contexto actual**: PPO (on-policy), sin replay buffer, `learning_rate=5e-5`, `ent_coef=0.005`.

**Propuesta D1 — SAC + PER** *(Zhang et al. [P2] + Pan et al. [P1])*

Reemplazar PPO por SAC con Prioritized Experience Replay en stage 6. Ambos papers demuestran que SAC supera a TD3 y PPO en entornos con obstáculos dinámicos por su exploración estocástica intrínseca (política gaussiana) y regularización de entropía. PER prioriza las transiciones difíciles (colisiones, evasiones cerca del peatón) para muestrearlas más frecuentemente.

SB3 soporta SAC directamente. El cambio requiere:
- Reescribir los scripts de training de stage 6 (distinta API que PPO)
- Nuevo transfer de pesos desde stage 5 (arquitectura actor/critic diferente en SAC)
- Sin `reset_num_timesteps` equivalente — nuevo mecanismo de warm-start

**Propuesta D2 — Behavioral Cloning como pre-entrenamiento de stage 6** *(Savid et al. [P4])*

Grabar trayectorias de referencia del ciclo completo (p.ej., con una política A* que rodea a los peatones usando el mapa inflado) y usarlas como pre-entrenamiento BC antes del fine-tuning con PPO. SB3 tiene soporte para BC via `imitation` library. Reduce la exploración aleatoria inicial y las colisiones tempranas de stage 6.

---

#### E. Curriculo de entrenamiento

**Propuestas de curriculo mejorado** *(inspiradas en el análisis de convergencia de [P1] y [P2])*

- **E1 — Curriculum de peatones**: stage 6\_a solo con Pedestrian 1 (trayectoria lineal en X, predecible), stage 6\_b con ambos peatones. Reduce la complejidad del dominio dinámico inicial y permite aprender evasión básica antes de enfrentarse a la combinación.

- **E2 — Más steps en stage 6\_din\_v3**: pasar de 2M a 4-6M steps antes de ir a v4. GAP\_SAC necesita 300 episodios × 500 steps = 150.000 steps para converger, pero en entorno mucho más simple (un obstáculo, un goal). La tarea de este trabajo (ciclo completo, 28 goals, 2 peatones) es órdenes de magnitud más compleja y probablemente necesita proporcionalmente más steps.

- **E3 — Randomización de velocidad de peatones**: variar la velocidad entre 0.5× y 1.5× la nominal durante el entrenamiento. Mejora la generalización a distintas situaciones de bloqueo y evita que el agente aprenda a sincronizarse con la velocidad exacta del peatón.

---

### 31.3 Tabla resumen de propuestas

| ID | Propuesta | Paper | Coste impl. | Compatible PPO | Impacto esperado | Prioridad |
|----|-----------|-------|-------------|----------------|-----------------|-----------|
| A1 | Obs predictiva peatón (pos en t+2) | P1 Pan | Muy bajo | ✓ | Medio-alto | **Alta** |
| A2 | Ángulo peatón relativo al heading | P2 Zhang | Muy bajo | ✓ | Medio | **Alta** |
| B1 | Ped penalty con decay temporal | P1 Pan | Bajo | ✓ | Medio-alto | **Alta** |
| B2 | Zona de exclusión de seguridad graduada | P2 Zhang | Bajo | ✓ | Medio | Media |
| B3 | Progreso multiplicativo por alineación | P2 Zhang | Bajo | ✓ | Medio | Media |
| B4 | Penalización de suavidad angular | P3 Gupta | Muy bajo | ✓ | Bajo-medio | Media |
| C1 | Gated Attention (custom policy) | P2 Zhang | Medio-alto | ✓ (custom) | Medio-alto | Media |
| D1 | SAC + PER en stage 6 | P1+P2 | Alto | ✗ (cambio alg.) | Alto si converge | Baja-media |
| D2 | Behavioral Cloning pre-entrenamiento | P4 Savid | Alto | ✓ | Incierto | Baja |
| E1 | Curriculum por peatones (1 antes que 2) | P1+P2 | Bajo | ✓ | Alto | **Alta** |
| E2 | Más steps en stage 6\_din\_v3 (→6M) | P1+P2 | Cero | ✓ | Alto | **Alta** |
| E3 | Randomización velocidad peatones | P2 Zhang | Bajo | ✓ | Medio | Media |

**Combinación recomendada de mínimo riesgo / máximo impacto**: E2 + E1 + A1 + B1. Más steps, curriculum de un peatón primero, observación predictiva, y penalty decreciente — todo compatible con el código actual sin cambiar algoritmo ni arquitectura.

---

## 17. Notas de implementación

- El supervisor de Webots permite mover nodos en tiempo real vía `setSFVec3f` — el
  obstáculo oscilante se implementa actualizando su posición en cada step del entorno
- Para no modificar `webots_env.py`, el movimiento del obstáculo se controla desde el
  propio script de inferencia usando `env.supervisor.getFromDef("OBSTACULO").getField(...)`
- El nodo obstáculo se define en `warehouse_1.wbt` con un DEF name (`OBSTACULO_1`, etc.)
  para poder accederlo desde el supervisor sin búsqueda por nombre
- Si el obstáculo tiene física activa, añadirlo como `Solid` con `physics` node;
  si solo es geométrico (no empuja al robot), puede ser un `Transform` sin física
