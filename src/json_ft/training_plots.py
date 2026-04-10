"""Training-history extraction and lightweight plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .utils import write_json


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render training curves. "
            "Install the train dependencies or use requirements-colab.txt."
        ) from exc
    return plt


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_training_history_payload(log_history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Convert trainer log history into a compact JSON-friendly structure."""

    entries = list(log_history)
    train_loss: list[dict[str, float]] = []
    eval_loss: list[dict[str, float]] = []
    learning_rate: list[dict[str, float]] = []

    for entry in entries:
        step = _coerce_float(entry.get("step"))
        epoch = _coerce_float(entry.get("epoch"))
        loss = _coerce_float(entry.get("loss"))
        eval_value = _coerce_float(entry.get("eval_loss"))
        lr = _coerce_float(entry.get("learning_rate"))

        if step is not None and loss is not None and eval_value is None:
            train_loss.append({"step": step, "epoch": epoch or 0.0, "loss": loss})
        if step is not None and eval_value is not None:
            eval_loss.append({"step": step, "epoch": epoch or 0.0, "eval_loss": eval_value})
        if step is not None and lr is not None:
            learning_rate.append({"step": step, "epoch": epoch or 0.0, "learning_rate": lr})

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "learning_rate": learning_rate,
        "raw_log_history": entries,
    }


def _render_curve(
    pyplot: Any,
    series: list[dict[str, float]],
    *,
    output_path: Path,
    title: str,
    value_key: str,
    color: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = pyplot.figure(figsize=(8, 4.5))
    steps = [point["step"] for point in series]
    values = [point[value_key] for point in series]
    pyplot.plot(steps, values, marker="o", linewidth=2, color=color)
    pyplot.title(title)
    pyplot.xlabel("Step")
    pyplot.ylabel(value_key.replace("_", " ").title())
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    figure.savefig(output_path, dpi=160)
    pyplot.close(figure)
    return output_path


def write_training_history_and_plots(
    *,
    log_history: Iterable[dict[str, Any]],
    history_path: str | Path,
    loss_curve_path: str | Path,
    eval_loss_curve_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist compact history JSON plus loss-curve images."""

    payload = build_training_history_payload(log_history)
    history_output = write_json(history_path, payload)

    plot_outputs: dict[str, str | None] = {
        "history_path": str(history_output),
        "loss_curve_path": None,
        "eval_loss_curve_path": None,
    }

    if payload["train_loss"]:
        pyplot = _load_pyplot()
        loss_output = _render_curve(
            pyplot,
            payload["train_loss"],
            output_path=Path(loss_curve_path).resolve(),
            title="SFT Training Loss",
            value_key="loss",
            color="#1f77b4",
        )
        plot_outputs["loss_curve_path"] = str(loss_output)

    if payload["eval_loss"] and eval_loss_curve_path is not None:
        pyplot = _load_pyplot()
        eval_output = _render_curve(
            pyplot,
            payload["eval_loss"],
            output_path=Path(eval_loss_curve_path).resolve(),
            title="SFT Evaluation Loss",
            value_key="eval_loss",
            color="#d62728",
        )
        plot_outputs["eval_loss_curve_path"] = str(eval_output)

    return plot_outputs
