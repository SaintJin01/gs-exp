"""A single positive/negative cosine training chart, with raw CSV history."""
import csv
import math
from pathlib import Path


class CosineHistory:
    fields = ("iteration", "camera_id", "positive_cosine", "negative_cosine")

    def __init__(self, output_dir, start_iteration=0):
        self.csv_path = Path(output_dir) / "cosine_history.csv"
        self.plot_path = Path(output_dir) / "cosine_similarity.png"
        self.rows = []
        if self.csv_path.is_file() and start_iteration > 0:
            with self.csv_path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(row["iteration"]) <= start_iteration:
                        self.rows.append({"iteration": int(row["iteration"]),
                                          "camera_id": int(row["camera_id"]),
                                          "positive_cosine": float(row["positive_cosine"]),
                                          "negative_cosine": float(row["negative_cosine"])})
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # A new training run starts a fresh history; resume retains past points
        # only up to the checkpoint iteration (no fabricated earlier values).
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def record(self, iteration, camera_id, positive, negative):
        row = dict(zip(self.fields, (iteration, camera_id, positive, negative)))
        self.rows.append(row)
        with self.csv_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(row)

    @staticmethod
    def smoothed(values, alpha=.05):
        result, average = [], None
        for value in values:
            if not math.isfinite(value):
                result.append(float("nan"))
                continue
            average = value if average is None else alpha * value + (1-alpha) * average
            result.append(average)
        return result

    def plot(self, alpha=.05, title=None, note=None):
        if not self.rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = [r["iteration"] for r in self.rows]
        for field, label, color in (("positive_cosine", "Positive (target: 1)", "tab:blue"),
                                     ("negative_cosine", "Negative (target: <= 0)", "tab:orange")):
            values = self.smoothed([r[field] for r in self.rows], alpha)
            ax.plot(iterations, values, color=color, label=label, linewidth=2,
                    marker="o" if len(values) == 1 else None)
        ax.axhline(0, color="gray", linewidth=.8, linestyle="--")
        ax.set(xlabel="Training iteration", ylabel="Mean cosine similarity",
               ylim=(-1.05, 1.05), title=title or "SAM feature training: positive / negative cosine")
        ax.grid(alpha=.2)
        ax.legend(loc="best")
        fig.text(.5, .015, note or "EMA alpha=0.05 | self-pairs excluded | sampled training views, not fixed validation",
                 ha="center", fontsize=9, color="gray")
        fig.tight_layout(rect=(0, .04, 1, 1))
        try:
            fig.savefig(str(self.plot_path), dpi=160)
        finally:
            plt.close(fig)
