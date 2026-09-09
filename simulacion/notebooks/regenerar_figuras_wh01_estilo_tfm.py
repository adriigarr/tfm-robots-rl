"""Regenera las figuras principales de la evaluación E2.1 en WH01.

Los CSV de inferencia son la única fuente de datos. El script modifica solo la
presentación gráfica y conserva el protocolo, las categorías y los valores.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figuras"

RESULT_COLORS = {
    "exito": "#2CA02C",
    "col_approach": "#D62728",
    "col_exit": "#FF7F0E",
    "truncado": "#7F7F7F",
}
RESULT_LABELS = {
    "exito": "Éxito",
    "col_approach": "Colisión en aproximación",
    "col_exit": "Colisión en salida",
    "truncado": "Truncamiento",
}
RESULT_ORDER = ("exito", "col_approach", "col_exit", "truncado")
GRID_COLOR = "#D9D9D9"

CONFIG = {
    "STH-WP": {
        "prefix": "sthwp",
        "directory": ROOT / "simulacion/controllers/rl_train_STHWP/experimentos/resultados",
    },
    "SUB-WP": {
        "prefix": "subwp",
        "directory": ROOT / "simulacion/controllers/rl_train_SUB_WP_continuo/experimentos/resultados",
    },
}


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


def style_axis(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)


def load_data(prefix: str, directory: Path) -> pd.DataFrame:
    frames = []
    for seed in (42, 123, 524):
        path = directory / f"{prefix}_infer_e2_1_s{seed}.csv"
        frame = pd.read_csv(path)
        frame["semilla"] = seed
        frame["resultado"] = frame["resultado"].replace(
            {"col_retorno": "col_exit", "col_return": "col_exit"}
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def render_step_distribution(architecture: str, prefix: str, frame: pd.DataFrame) -> Path:
    outcomes = [outcome for outcome in RESULT_ORDER if outcome in set(frame["resultado"])]
    values = [frame.loc[frame["resultado"] == outcome, "pasos"].dropna().to_numpy() for outcome in outcomes]
    labels = [RESULT_LABELS[outcome] for outcome in outcomes]
    colors = [RESULT_COLORS[outcome] for outcome in outcomes]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    boxes = axes[0].boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in boxes["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.8)

    violins = axes[1].violinplot(values, showmeans=True, showmedians=True, showextrema=True)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in violins:
            violins[key].set_color("#333333")
            violins[key].set_linewidth(1.0)
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)

    axes[0].set_title("Diagrama de caja")
    axes[1].set_title("Diagrama de violín")
    for ax in axes:
        ax.set_xlabel("Resultado del episodio")
        ax.tick_params(axis="x", rotation=16)
        style_axis(ax, axis="y")
    axes[0].set_ylabel("Pasos por episodio")
    fig.suptitle(f"{architecture}: distribución de la duración de los episodios en E2.1")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = FIG_DIR / f"fig_{prefix}_e2_1_08_pasos_distribucion.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def render_outcomes_by_goal(architecture: str, prefix: str, frame: pd.DataFrame) -> Path:
    goals = sorted(
        frame["goal_id"].dropna().unique(),
        key=lambda goal: int(str(goal).split("_")[-1]),
    )
    outcomes = [outcome for outcome in RESULT_ORDER if outcome in set(frame["resultado"])]
    totals = frame.groupby("goal_id").size()
    x = np.arange(len(goals))
    bottom = np.zeros(len(goals))

    fig, ax = plt.subplots(figsize=(15, 5.7))
    for outcome in outcomes:
        counts = frame[frame["resultado"] == outcome].groupby("goal_id").size()
        percentages = np.array([100.0 * counts.get(goal, 0) / totals[goal] for goal in goals])
        ax.bar(
            x,
            percentages,
            bottom=bottom,
            color=RESULT_COLORS[outcome],
            label=RESULT_LABELS[outcome],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
        )
        bottom += percentages

    ax.set_xticks(x)
    ax.set_xticklabels([str(goal).replace("goal_", "G") for goal in goals], rotation=45, ha="right")
    ax.set_ylabel("Proporción de episodios (%)")
    ax.set_xlabel("Ubicación de recogida")
    ax.set_ylim(0, 100)
    ax.set_title(f"{architecture}: distribución de resultados por ubicación en E2.1")
    ax.legend(ncol=min(4, len(outcomes)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style_axis(ax, axis="y")
    fig.tight_layout()

    output = FIG_DIR / f"fig_{prefix}_e2_1_infer_stacked.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for architecture, config in CONFIG.items():
        frame = load_data(config["prefix"], config["directory"])
        outputs.append(render_step_distribution(architecture, config["prefix"], frame))
        outputs.append(render_outcomes_by_goal(architecture, config["prefix"], frame))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
