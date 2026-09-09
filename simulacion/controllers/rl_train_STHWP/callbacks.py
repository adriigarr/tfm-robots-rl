import csv
import os
from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback

# Índices 0-based de estanterías difíciles (pasillo estrecho con giro 90°)
SHELVES_DIFICILES_IDX = {4, 8, 9, 12, 16, 17, 18, 22, 23, 24}


class StatsCallback(BaseCallback):
    """
    Callback de métricas para el entrenamiento STH-WP en 4 etapas.

    TensorBoard — stats/ (por episodio, ventana deslizante):
      tasa_exito_%               Éxito acumulado
      tasa_colision_%            Colisión acumulada
      tasa_truncado_%            Truncado acumulado
      tasa_estanteria_%          Llegada a estantería acumulada
      exito_ult{W}_%             Éxito ventana deslizante
      colision_ult{W}_%          Colisión ventana deslizante
      truncado_ult{W}_%          Truncado ventana deslizante
      reward_medio_episodio      Reward acumulado medio (ventana)
      pasos_medio_episodio       Pasos por episodio (ventana)
      exito_dificiles_ult{W}_%   Éxito en goals difíciles (ventana)
      exito_faciles_ult{W}_%     Éxito en goals fáciles (ventana)

    TensorBoard — stage4/ (solo etapa 4, approach vs exit):
      llego_estanteria_ult{W}_%  Approach completado (ventana)
      colision_approach_%        Colisión durante approach (acumulada)
      colision_exit_%            Colisión durante exit (acumulada)
      truncado_approach_%        Truncado durante approach (acumulada)
      truncado_exit_%            Truncado durante exit (acumulada)

    TensorBoard — goal/ (cada LOG_GOAL_FREQ episodios):
      goal_{id}_exito_%          Tasa de éxito por goal

    CSV al finalizar: stats_stage{N}.csv
    """

    VENTANA        = 100
    LOG_GOAL_FREQ  = 200   # cada cuántos episodios loggear métricas por goal

    def __init__(self, goal_ids, stage=1, run_id="run001", verbose=0):
        super().__init__(verbose)
        self.goal_ids = goal_ids
        self.stage    = stage
        self.run_id   = run_id

        # IDs de goals difíciles (string) para el split
        self._ids_dificiles = {
            goal_ids[i] for i in SHELVES_DIFICILES_IDX if i < len(goal_ids)
        }

        # ── contadores globales ───────────────────────────────────────────────
        self.n_episodios  = 0
        self.n_exitos     = 0
        self.n_colisiones = 0
        self.n_truncados  = 0
        self.n_estanteria = 0

        # ── ventanas deslizantes globales ─────────────────────────────────────
        self._w_exito    = deque(maxlen=self.VENTANA)
        self._w_colision = deque(maxlen=self.VENTANA)
        self._w_truncado = deque(maxlen=self.VENTANA)
        self._w_pasos    = deque(maxlen=self.VENTANA)
        self._w_reward   = deque(maxlen=self.VENTANA)

        # ── ventanas por dificultad ───────────────────────────────────────────
        self._w_exito_dif   = deque(maxlen=self.VENTANA)
        self._w_exito_facil = deque(maxlen=self.VENTANA)

        # ── stage 4: desglose approach / exit ────────────────────────────────
        self.n_llego_estanteria  = 0
        self._w_llego_estanteria = deque(maxlen=self.VENTANA)
        self.n_col_approach  = 0
        self.n_col_exit      = 0
        self.n_trunc_approach = 0
        self.n_trunc_exit    = 0

        # ── por goal ──────────────────────────────────────────────────────────
        self.goal_intentos   = defaultdict(int)
        self.goal_exitos     = defaultdict(int)
        self.goal_colisiones = defaultdict(int)
        self.goal_estanteria = defaultdict(int)

        # ── acumulador de reward ──────────────────────────────────────────────
        self._reward_episodio = 0.0
        self._pasos_episodio  = 0

    def _get_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    def _on_step(self) -> bool:
        self._pasos_episodio  += 1
        self._reward_episodio += float(self.locals["rewards"][0])

        if not self.locals["dones"][0]:
            return True

        # ── fin de episodio ───────────────────────────────────────────────────
        self.n_episodios += 1
        info     = self.locals["infos"][0]
        base_env = self._get_env()

        goal_idx = base_env.current_goal_idx
        goal_id  = self.goal_ids[goal_idx]

        es_exito         = bool(info.get("exito",            False))
        es_colision      = bool(info.get("colision",         False))
        es_truncado      = bool(info.get("truncado",         False))
        llego_estanteria = bool(info.get("llego_estanteria", False))

        # ── contadores globales ───────────────────────────────────────────────
        self.goal_intentos[goal_id] += 1
        if llego_estanteria:
            self.n_estanteria += 1
            self.goal_estanteria[goal_id] += 1
        if es_exito:
            self.n_exitos += 1
            self.goal_exitos[goal_id] += 1
        elif es_colision:
            self.n_colisiones += 1
            self.goal_colisiones[goal_id] += 1
        else:
            self.n_truncados += 1

        # ── ventanas globales ─────────────────────────────────────────────────
        self._w_exito.append(1 if es_exito else 0)
        self._w_colision.append(1 if es_colision else 0)
        self._w_truncado.append(1 if es_truncado else 0)
        self._w_pasos.append(self._pasos_episodio)
        self._w_reward.append(self._reward_episodio)

        # ── ventanas por dificultad ───────────────────────────────────────────
        if goal_id in self._ids_dificiles:
            self._w_exito_dif.append(1 if es_exito else 0)
        else:
            self._w_exito_facil.append(1 if es_exito else 0)

        # ── stage 4: desglose approach / exit ────────────────────────────────
        if self.stage == 4:
            self._w_llego_estanteria.append(1 if llego_estanteria else 0)
            if llego_estanteria:
                self.n_llego_estanteria += 1
            if es_colision:
                if llego_estanteria:
                    self.n_col_exit += 1
                else:
                    self.n_col_approach += 1
            if es_truncado:
                if llego_estanteria:
                    self.n_trunc_exit += 1
                else:
                    self.n_trunc_approach += 1

        # ── reset acumuladores de episodio ────────────────────────────────────
        self._reward_episodio = 0.0
        self._pasos_episodio  = 0

        # ── log TensorBoard ───────────────────────────────────────────────────
        N = self.n_episodios

        # tasas acumuladas
        self.logger.record("stats/tasa_exito_%",       self.n_exitos     / N * 100)
        self.logger.record("stats/tasa_colision_%",    self.n_colisiones / N * 100)
        self.logger.record("stats/tasa_truncado_%",    self.n_truncados  / N * 100)
        self.logger.record("stats/tasa_estanteria_%",  self.n_estanteria / N * 100)
        self.logger.record("stats/n_exitos",           self.n_exitos)
        self.logger.record("stats/n_colisiones",       self.n_colisiones)
        self.logger.record("stats/n_truncados",        self.n_truncados)

        # ventanas deslizantes (disponibles tras ≥10 episodios)
        if len(self._w_exito) >= 10:
            W = self.VENTANA
            self.logger.record(f"stats/exito_ult{W}_%",
                               sum(self._w_exito)    / len(self._w_exito)    * 100)
            self.logger.record(f"stats/colision_ult{W}_%",
                               sum(self._w_colision) / len(self._w_colision) * 100)
            self.logger.record(f"stats/truncado_ult{W}_%",
                               sum(self._w_truncado) / len(self._w_truncado) * 100)
            self.logger.record("stats/reward_medio_episodio",
                               sum(self._w_reward)   / len(self._w_reward))
            self.logger.record("stats/pasos_medio_episodio",
                               sum(self._w_pasos)    / len(self._w_pasos))

        if len(self._w_exito_dif) >= 5:
            self.logger.record(f"stats/exito_dificiles_ult{self.VENTANA}_%",
                               sum(self._w_exito_dif) / len(self._w_exito_dif) * 100)
        if len(self._w_exito_facil) >= 5:
            self.logger.record(f"stats/exito_faciles_ult{self.VENTANA}_%",
                               sum(self._w_exito_facil) / len(self._w_exito_facil) * 100)

        # stage 4: approach vs exit
        if self.stage == 4 and len(self._w_llego_estanteria) >= 10:
            W = self.VENTANA
            self.logger.record(f"stage4/llego_estanteria_ult{W}_%",
                               sum(self._w_llego_estanteria) / len(self._w_llego_estanteria) * 100)
            if self.n_episodios > 0:
                self.logger.record("stage4/colision_approach_%",
                                   self.n_col_approach  / N * 100)
                self.logger.record("stage4/colision_exit_%",
                                   self.n_col_exit      / N * 100)
                self.logger.record("stage4/truncado_approach_%",
                                   self.n_trunc_approach / N * 100)
                self.logger.record("stage4/truncado_exit_%",
                                   self.n_trunc_exit    / N * 100)

        # métricas por goal (periódicas)
        if self.n_episodios % self.LOG_GOAL_FREQ == 0:
            for gid in self.goal_ids:
                intentos = self.goal_intentos[gid]
                if intentos > 0:
                    self.logger.record(
                        f"goal/{gid}_exito_%",
                        self.goal_exitos[gid] / intentos * 100,
                    )

        # print consola cada 50 episodios
        if self.n_episodios % 50 == 0:
            vent_ex  = sum(self._w_exito)    / max(len(self._w_exito),    1) * 100
            vent_col = sum(self._w_colision) / max(len(self._w_colision), 1) * 100
            vent_dif = sum(self._w_exito_dif)   / max(len(self._w_exito_dif),   1) * 100
            vent_fac = sum(self._w_exito_facil) / max(len(self._w_exito_facil), 1) * 100
            pasos_m  = sum(self._w_pasos)    / max(len(self._w_pasos),    1)
            reward_m = sum(self._w_reward)   / max(len(self._w_reward),   1)
            print(
                f"\n[STATS ep={self.n_episodios} stage={self.stage}] "
                f"Éxito últ{self.VENTANA}: {vent_ex:.1f}% | "
                f"Col: {vent_col:.1f}% | "
                f"Dif: {vent_dif:.1f}% | "
                f"Fácil: {vent_fac:.1f}% | "
                f"Pasos: {pasos_m:.0f} | "
                f"Reward: {reward_m:.1f}"
            )

        return True

    def _on_training_end(self):
        csv_path = f"stats_{self.run_id}_stage{self.stage}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "goal_id", "dificil", "intentos", "exitos", "colisiones",
                "llego_estanteria", "tasa_exito_%", "tasa_estanteria_%",
            ])
            for goal_id in self.goal_ids:
                intentos   = self.goal_intentos[goal_id]
                exitos     = self.goal_exitos[goal_id]
                colisiones = self.goal_colisiones[goal_id]
                estanteria = self.goal_estanteria[goal_id]
                dificil    = goal_id in self._ids_dificiles
                tasa_ex    = (exitos     / intentos * 100) if intentos > 0 else 0.0
                tasa_est   = (estanteria / intentos * 100) if intentos > 0 else 0.0
                writer.writerow([
                    goal_id, dificil, intentos, exitos, colisiones,
                    estanteria, f"{tasa_ex:.1f}", f"{tasa_est:.1f}",
                ])
        print(f"\n[CSV] Estadísticas {self.run_id} stage {self.stage} guardadas en {csv_path}")
