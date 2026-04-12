"""Training-history extraction and lightweight plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
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


def _slugify_metric_key(metric_key: str) -> str:
    """Convert metric keys like `rewards/chosen` into safe dict/file labels."""

    return metric_key.replace("/", "_").replace(" ", "_")


@dataclass(frozen=True)
class PlotSpec:
    """Describe one scalar series plot that should be rendered from trainer logs."""

    metric_key: str
    output_path: Path
    title: str
    color: str


def build_training_history_payload(
    log_history: Iterable[dict[str, Any]],
    tracked_scalar_keys: Iterable[str] | None = None,
    derived_scalar_series: dict[str, list[dict[str, float]]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert trainer log history into a compact JSON-friendly structure."""

    entries = list(log_history)
    train_loss: list[dict[str, float]] = []
    eval_loss: list[dict[str, float]] = []
    learning_rate: list[dict[str, float]] = []
    requested_keys = []
    if tracked_scalar_keys is not None:
        requested_keys = [str(key) for key in tracked_scalar_keys if str(key).strip()]

    tracked_keys = list(dict.fromkeys(["loss", "eval_loss", "learning_rate", *requested_keys]))
    scalar_series: dict[str, list[dict[str, float]]] = {key: [] for key in tracked_keys}

    for entry in entries:
        step = _coerce_float(entry.get("step"))
        epoch = _coerce_float(entry.get("epoch"))
        loss = _coerce_float(entry.get("loss"))
        if loss is None:
            # TRL DPO logs often emit `train_loss` instead of `loss`.
            loss = _coerce_float(entry.get("train_loss"))
        eval_value = _coerce_float(entry.get("eval_loss"))
        lr = _coerce_float(entry.get("learning_rate"))

        if step is not None:
            for metric_key in tracked_keys:
                if metric_key == "loss":
                    value = loss
                else:
                    value = _coerce_float(entry.get(metric_key))
                if value is None:
                    continue
                scalar_series[metric_key].append(
                    {
                        "step": step,
                        "epoch": epoch or 0.0,
                        metric_key: value,
                    }
                )

        if step is not None and loss is not None and eval_value is None:
            train_loss.append({"step": step, "epoch": epoch or 0.0, "loss": loss})
        if step is not None and eval_value is not None:
            eval_loss.append({"step": step, "epoch": epoch or 0.0, "eval_loss": eval_value})
        if step is not None and lr is not None:
            learning_rate.append({"step": step, "epoch": epoch or 0.0, "learning_rate": lr})

    payload = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "learning_rate": learning_rate,
        "scalar_series": scalar_series,
        "raw_log_history": entries,
    }
    for metric_key, series in (derived_scalar_series or {}).items():
        payload["scalar_series"][metric_key] = list(series)
    if metadata:
        payload["metadata"] = metadata
    return payload


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


def render_scalar_series_plots(
    *,
    history_payload: dict[str, Any],
    plot_specs: Iterable[PlotSpec],
) -> dict[str, str]:
    """Render plots for any tracked scalar series present in the history payload."""

    plot_outputs: dict[str, str] = {}
    scalar_series = history_payload.get("scalar_series", {})
    pyplot: Any | None = None
    for spec in plot_specs:
        series = list(scalar_series.get(spec.metric_key, []))
        if not series:
            continue
        pyplot = pyplot or _load_pyplot()
        rendered = _render_curve(
            pyplot,
            series,
            output_path=spec.output_path,
            title=spec.title,
            value_key=spec.metric_key,
            color=spec.color,
        )
        plot_outputs[_slugify_metric_key(spec.metric_key)] = str(rendered)
    return plot_outputs


def write_training_history_and_plots(
    *,
    log_history: Iterable[dict[str, Any]],
    history_path: str | Path,
    loss_curve_path: str | Path,
    eval_loss_curve_path: str | Path | None = None,
    tracked_scalar_keys: Iterable[str] | None = None,
    extra_plot_specs: Iterable[PlotSpec] | None = None,
    derived_scalar_series: dict[str, list[dict[str, float]]] | None = None,
    metadata: dict[str, Any] | None = None,
    loss_curve_title: str = "SFT Training Loss",
    eval_loss_curve_title: str = "SFT Evaluation Loss",
) -> dict[str, Any]:
    """Persist compact history JSON plus loss-curve images."""

    payload = build_training_history_payload(
        log_history,
        tracked_scalar_keys=tracked_scalar_keys,
        derived_scalar_series=derived_scalar_series,
        metadata=metadata,
    )
    history_output = write_json(history_path, payload)

    plot_outputs: dict[str, str | None] = {
        "history_path": str(history_output),
        "loss_curve_path": None,
        "eval_loss_curve_path": None,
        "extra_plot_paths": {},
    }

    if payload["train_loss"]:
        pyplot = _load_pyplot()
        loss_output = _render_curve(
            pyplot,
            payload["train_loss"],
            output_path=Path(loss_curve_path).resolve(),
            title=loss_curve_title,
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
            title=eval_loss_curve_title,
            value_key="eval_loss",
            color="#d62728",
        )
        plot_outputs["eval_loss_curve_path"] = str(eval_output)

    if extra_plot_specs:
        plot_outputs["extra_plot_paths"] = render_scalar_series_plots(
            history_payload=payload,
            plot_specs=extra_plot_specs,
        )

    return plot_outputs
