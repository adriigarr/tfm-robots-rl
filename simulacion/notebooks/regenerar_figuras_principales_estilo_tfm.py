"""Regenera las figuras principales del entrenamiento con un estilo uniforme.

El script no modifica ni recalcula los resultados experimentales. Lee los
escalares ya registrados en TensorBoard y los CSV agregados que produjo el
notebook de análisis, y vuelve a representar únicamente las cuatro figuras que
se emplean en el cuerpo principal de la memoria.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
RESULT_DIR = NOTEBOOK_DIR / "resultados_entrenamiento_hasta_e2_1"
FIG_DIR = NOTEBOOK_DIR / "figuras_entrenamiento_hasta_e2_1"

SEEDS = (42, 123, 524)
SEED_COLORS = {42: "#1F77B4", 123: "#E6A700", 524: "#2CA02C"}
GRID_COLOR = "#D9D9D9"

RUNS = {
    "STH-WP": {
        42: ROOT / "simulacion/controllers/rl_train_STHWP/tensorboard_logs/sthwp_e2_1_s42_1",
        123: ROOT / "simulacion/controllers/rl_train_STHWP/tensorboard_logs/sthwp_e2_1_s123_1",
        524: ROOT / "simulacion/controllers/rl_train_STHWP/tensorboard_logs/sthwp_e2_1_s524_1",
    },
    "SUB-WP": {
        42: ROOT / "simulacion/controllers/rl_train_SUB_WP_continuo/tensorboard_logs/subwp_e2_1_s42_1",
        123: ROOT / "simulacion/controllers/rl_train_SUB_WP_continuo/tensorboard_logs/subwp_e2_1_s123_2",
        524: ROOT / "simulacion/controllers/rl_train_SUB_WP_continuo/tensorboard_logs/subwp_e2_1_s524_1",
    },
}

FUNCTIONAL_METRICS = (
    ("stats/tasa_exito_%", "Tasa de éxito (%)"),
    ("stats/tasa_colision_%", "Tasa de colisión (%)"),
    ("stats/tasa_truncado_%", "Tasa de truncamiento (%)"),
    ("rollout/ep_rew_mean", "Recompensa media por episodio"),
    ("rollout/ep_len_mean", "Longitud media del episodio (pasos)"),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)


def ema(values: pd.Series, alpha: float = 0.85) -> pd.Series:
    return values.ewm(alpha=1.0 - alpha, adjust=False).mean()


def load_scalars(run_dir: Path, tags: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    result: dict[str, pd.DataFrame] = {}
    for tag in tags:
        if tag not in available:
            result[tag] = pd.DataFrame(columns=["step", "value"])
            continue
        events = accumulator.Scalars(tag)
        result[tag] = pd.DataFrame(
            {"step": [event.step for event in events], "value": [event.value for event in events]}
        )
    return result


def render_functional_metrics(architecture: str) -> Path:
    tags = tuple(tag for tag, _ in FUNCTIONAL_METRICS)
    data = {seed: load_scalars(RUNS[architecture][seed], tags) for seed in SEEDS}

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.2))
    axes = axes.ravel()
    for index, (tag, label) in enumerate(FUNCTIONAL_METRICS):
        ax = axes[index]
        for seed in SEEDS:
            frame = data[seed][tag]
            if frame.empty:
                continue
            color = SEED_COLORS[seed]
            ax.plot(frame["step"], frame["value"], color=color, linewidth=0.8, alpha=0.16)
            ax.plot(frame["step"], ema(frame["value"]), color=color, linewidth=1.8, alpha=0.98)
        ax.set_title(label)
        ax.set_xlabel("Pasos de entrenamiento en E2.1")
        ax.set_ylabel(label)
        style_axis(ax)
    fig.delaxes(axes[-1])

    handles = [
        Line2D([0], [0], color=SEED_COLORS[seed], linewidth=2, label=f"Semilla {seed}")
        for seed in SEEDS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{architecture}: métricas funcionales del entrenamiento E2.1", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    prefix = "sth_wp" if architecture == "STH-WP" else "sub_wp"
    output = FIG_DIR / f"{prefix}_e2_1_funcionales.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def render_curriculum(architecture: str) -> Path:
    prefix = "sth_wp" if architecture == "STH-WP" else "sub_wp"
    frame = pd.read_csv(RESULT_DIR / f"{prefix}_evolucion_curricular.csv")
    phase_order = (
        frame[["fase", "orden"]]
        .drop_duplicates()
        .sort_values("orden")["fase"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for seed in SEEDS:
        subset = frame[frame["semilla"] == seed].set_index("fase").reindex(phase_order)
        axes[0].plot(
            phase_order,
            subset["exito"],
            color=SEED_COLORS[seed],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"Semilla {seed}",
        )
        axes[1].plot(
            phase_order,
            subset["colision"],
            color=SEED_COLORS[seed],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"Semilla {seed}",
        )

    axes[0].set_title("Tasa de éxito al final de cada fase")
    axes[0].set_ylabel("Tasa de éxito (%)")
    axes[1].set_title("Tasa de colisión al final de cada fase")
    axes[1].set_ylabel("Tasa de colisión (%)")
    for ax in axes:
        ax.set_xlabel("Fase de entrenamiento")
        ax.tick_params(axis="x", rotation=45)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{architecture}: evolución resumida del entrenamiento progresivo", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output = FIG_DIR / f"{prefix}_evolucion_curricular_resumida.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum-only",
        action="store_true",
        help="Regenera solo los resúmenes curriculares a partir de los CSV agregados.",
    )
    args = parser.parse_args()
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for architecture in RUNS:
        if not args.curriculum_only:
            outputs.append(render_functional_metrics(architecture))
        outputs.append(render_curriculum(architecture))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
