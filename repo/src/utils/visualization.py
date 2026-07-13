"""
visualization.py
Genera automaticamente tablas comparativas, graficos, curvas PR y
matrices de confusion, y los guarda en results/tables y results/plots.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import TABLES_DIR, PLOTS_DIR


def save_comparison_table(results_dict: dict, filename: str = "comparison_table") -> pd.DataFrame:
    """
    results_dict: {model_name: {metric_name: value, ...}, ...}
    Guarda CSV, JSON y una version Markdown de la tabla comparativa.
    """
    df = pd.DataFrame(results_dict).T

    csv_path = os.path.join(TABLES_DIR, f"{filename}.csv")
    json_path = os.path.join(TABLES_DIR, f"{filename}.json")
    md_path = os.path.join(TABLES_DIR, f"{filename}.md")

    df.to_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    with open(md_path, "w") as f:
        f.write(df.to_markdown())

    print(f"[visualization] Tabla guardada en {csv_path}, {json_path}, {md_path}")
    return df


def plot_bar_comparison(results_dict: dict, metric_key: str, title: str, ylabel: str,
                         filename: str) -> None:
    """Grafico de barras comparando un metric_key entre modelos."""
    models = list(results_dict.keys())
    values = [results_dict[m][metric_key] for m in models]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=models, y=values, hue=models, palette="viridis", legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualization] Grafico guardado en {out_path}")


def plot_confusion_matrix(cm: list, labels: list, title: str, filename: str) -> None:
    """Matriz de confusion 2x2 para la metrica de nivel-evento."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prediccion")
    plt.ylabel("Ground truth")
    plt.title(title)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualization] Matriz de confusion guardada en {out_path}")


def plot_precision_recall_curve(precisions, recalls, title: str, filename: str) -> None:
    """Curva Precision-Recall para una clase."""
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualization] Curva PR guardada en {out_path}")


def plot_prompt_sensitivity(sensitivity_dict: dict, filename: str = "prompt_sensitivity") -> None:
    """Grafico de barras del F1 por variante de prompt (robustez del VLM)."""
    prompts = list(sensitivity_dict["per_prompt"].keys())
    f1_scores = list(sensitivity_dict["per_prompt"].values())
    short_labels = [f"V{i+1}" for i in range(len(prompts))]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=short_labels, y=f1_scores, hue=short_labels, palette="magma", legend=False)
    plt.axhline(sensitivity_dict["mean_f1"], color="red", linestyle="--", label="F1 promedio")
    plt.ylabel("F1 nivel-evento")
    plt.title("Sensibilidad al prompt (VLM generativo)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{filename}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualization] Grafico de sensibilidad guardado en {out_path}")
