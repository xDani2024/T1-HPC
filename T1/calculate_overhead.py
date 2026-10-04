"""Calcula y grafica el overhead paralelo del item (f)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("resultados")
INPUT_FILES = {
    "Apple Silicon": RESULTS_DIR / "resultados_f_chea.csv",
    "Intel/WSL2": RESULTS_DIR / "resultados_f_coni.csv",
}
OUTPUT_CSV = RESULTS_DIR / "overhead_f.csv"
OUTPUT_PNG = RESULTS_DIR / "overhead_f.png"
IMPLEMENTATIONS = ("numpy", "sklearn", "auto")
LABELS = {
    "numpy": "NumPy",
    "sklearn": "sklearn",
    "auto": "BaggingRegressor",
}


def read_measurements(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def calculate_overhead(rows):
    baseline = {
        name: float(rows[0][name])
        for name in IMPLEMENTATIONS
    }
    calculated = []

    for row in rows:
        p = int(row["p"])
        result = {"p": p}

        for name in IMPLEMENTATIONS:
            time_p = float(row[name])
            overhead = p * time_p - baseline[name]
            result[f"{name}_time"] = time_p
            result[f"{name}_overhead"] = overhead
            result[f"{name}_relative"] = overhead / baseline[name]

        calculated.append(result)

    return calculated


def save_csv(all_results):
    fields = ["machine", "p"]
    for name in IMPLEMENTATIONS:
        fields.extend(
            [f"{name}_time", f"{name}_overhead", f"{name}_relative"]
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for machine, rows in all_results.items():
            for row in rows:
                writer.writerow(
                    {"machine": machine, **row}
                )


def plot(all_results):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    for axis, (machine, rows) in zip(axes, all_results.items()):
        p_values = [row["p"] for row in rows]

        for name in IMPLEMENTATIONS:
            overheads = [row[f"{name}_overhead"] for row in rows]
            axis.plot(
                p_values,
                overheads,
                marker="o",
                linewidth=2,
                label=LABELS[name],
            )

        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(machine)
        axis.set_xlabel("Procesos p")
        axis.set_xticks(p_values)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel(r"Overhead $T_o(p)=pT(p)-T(1)$ [s]")
    axes[1].legend()
    figure.suptitle("Overhead paralelo del bootstrap (B=48, t=1)")
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=160)
    plt.close(figure)


def print_summary(all_results):
    for machine, rows in all_results.items():
        print(f"\n{machine}")
        for name in IMPLEMENTATIONS:
            best = min(rows, key=lambda row: row[f"{name}_time"])
            overhead = best[f"{name}_overhead"]
            relative = best[f"{name}_relative"] * 100
            print(
                f"  {LABELS[name]}: mejor p={best['p']}, "
                f"T={best[f'{name}_time']:.2f} s, "
                f"To={overhead:.2f} s ({relative:.1f}%)"
            )


def main():
    all_results = {
        machine: calculate_overhead(read_measurements(path))
        for machine, path in INPUT_FILES.items()
    }
    save_csv(all_results)
    plot(all_results)
    print_summary(all_results)
    print(f"\nTabla: {OUTPUT_CSV}")
    print(f"Grafico: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()