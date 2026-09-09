import csv
import os
from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback


class StatsCallback(BaseCallback):

    VENTANA = 100  # episodios para la tasa deslizante

    def __init__(self, goal_ids, verbose=0):
        super().__init__(verbose)
        self.goal_ids = goal_ids

        # contadores globales acumulados
        self.n_episodios  = 0
        self.n_exitos     = 0
        self.n_colisiones = 0
        self.n_truncados  = 0
        self.n_estanteria = 0   # llegó a la estantería (aunque luego fallara)

        # ventana deslizante para métricas recientes
        self._ventana_exito    = deque(maxlen=self.VENTANA)
        self._ventana_colision = deque(maxlen=self.VENTANA)
        self._ventana_pasos    = deque(maxlen=self.VENTANA)

        # contadores por goal
        self.goal_intentos   = defaultdict(int)
        self.goal_exitos     = defaultdict(int)
        self.goal_colisiones = defaultdict(int)
        self.goal_estanteria = defaultdict(int)

        # longitud del episodio actual
        self._pasos_episodio = 0

    def _get_env(self):
        """Devuelve el entorno base de forma robusta."""
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    def _on_step(self) -> bool:
        self._pasos_episodio += 1

        if self.locals["dones"][0]:
            self.n_episodios += 1
            info     = self.locals["infos"][0]
            base_env = self._get_env()

            goal_idx = base_env.current_goal_idx
            goal_id  = self.goal_ids[goal_idx]

            # leer llego_estanteria desde info (fiable tras el fix)
            llego_estanteria = bool(info.get("llego_estanteria", False))

            self.goal_intentos[goal_id] += 1

            # ── clasificar episodio ───────────────────────────────────────────
            es_exito    = bool(info.get("exito"))
            es_colision = bool(info.get("colision"))

            if es_exito:
                self.n_exitos += 1
                self.goal_exitos[goal_id] += 1
            elif es_colision:
                self.n_colisiones += 1
                self.goal_colisiones[goal_id] += 1
            else:
                self.n_truncados += 1

            if llego_estanteria:
                self.n_estanteria += 1
                self.goal_estanteria[goal_id] += 1

            # ── ventana deslizante ────────────────────────────────────────────
            self._ventana_exito.append(1 if es_exito else 0)
            self._ventana_colision.append(1 if es_colision else 0)
            self._ventana_pasos.append(self._pasos_episodio)
            self._pasos_episodio = 0

            # ── métricas acumuladas ───────────────────────────────────────────
            tasa_exito    = self.n_exitos     / self.n_episodios * 100
            tasa_colision = self.n_colisiones / self.n_episodios * 100
            tasa_estant   = self.n_estanteria / self.n_episodios * 100

            self.logger.record("stats/tasa_exito_%",         tasa_exito)
            self.logger.record("stats/tasa_colision_%",      tasa_colision)
            self.logger.record("stats/tasa_estanteria_%",    tasa_estant)
            self.logger.record("stats/n_colisiones",         self.n_colisiones)
            self.logger.record("stats/n_exitos",             self.n_exitos)
            self.logger.record("stats/n_truncados",          self.n_truncados)

            # ── métricas ventana deslizante (más útiles al principio) ─────────
            if len(self._ventana_exito) >= 10:
                self.logger.record(
                    f"stats/exito_ultimos_{self.VENTANA}_%",
                    sum(self._ventana_exito)    / len(self._ventana_exito) * 100
                )
                self.logger.record(
                    f"stats/colision_ultimos_{self.VENTANA}_%",
                    sum(self._ventana_colision) / len(self._ventana_colision) * 100
                )
                self.logger.record(
                    "stats/pasos_medio_episodio",
                    sum(self._ventana_pasos)    / len(self._ventana_pasos)
                )

            if self.n_episodios % 50 == 0:
                vent_ex  = sum(self._ventana_exito)    / max(len(self._ventana_exito), 1) * 100
                vent_col = sum(self._ventana_colision) / max(len(self._ventana_colision), 1) * 100
                pasos_m  = sum(self._ventana_pasos)    / max(len(self._ventana_pasos), 1)
                print(
                    f"\n[STATS ep={self.n_episodios}] "
                    f"Éxito acum: {tasa_exito:.1f}% | "
                    f"Éxito últ.{self.VENTANA}: {vent_ex:.1f}% | "
                    f"Colisión últ.{self.VENTANA}: {vent_col:.1f}% | "
                    f"Estantería: {tasa_estant:.1f}% | "
                    f"Pasos/ep: {pasos_m:.0f}"
                )

        return True

    def _on_training_end(self):
        csv_path = "stats_por_goal_sthwp_17.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "goal_id", "intentos", "exitos", "colisiones",
                "llego_estanteria", "tasa_exito_%", "tasa_estanteria_%"
            ])
            for goal_id in self.goal_ids:
                intentos   = self.goal_intentos[goal_id]
                exitos     = self.goal_exitos[goal_id]
                colisiones = self.goal_colisiones[goal_id]
                estanteria = self.goal_estanteria[goal_id]
                tasa_ex    = (exitos     / intentos * 100) if intentos > 0 else 0.0
                tasa_est   = (estanteria / intentos * 100) if intentos > 0 else 0.0
                writer.writerow([
                    goal_id, intentos, exitos, colisiones,
                    estanteria, f"{tasa_ex:.1f}", f"{tasa_est:.1f}"
                ])
        print(f"\n[CSV] Estadísticas guardadas en {csv_path}")