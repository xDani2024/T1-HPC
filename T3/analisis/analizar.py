"""Analisis de resultados de la Tarea 3 (IIC3533).

Uso (desde la carpeta T3):  python3 analisis/analizar.py
Lee:  resultados_lu_serial.txt (parte a), resultados_cluster/*.csv (partes c-f),
      logs/lu_prueba_entorno_*.out (correctitud en MPI real, parte b)
Genera en analisis/salida/: figuras PNG y tablas (CSV + tablas.md listo para el informe).

Convenciones:
- Una "corrida" = una ejecucion de mpirun (identificada por archivo, fecha, rep, b).
- tiempo de la corrida = max_rank(total)  (tiempo_global_s).
- computo/comunicacion de una corrida = promedio y maximo entre ranks.
- Cada configuracion se resume con la MEDIANA de sus repeticiones.
- Speedup S = t(config 0, T=1, mismo b) / t ; eficiencia E = S / CPUs totales (p*T).
"""
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "analisis" / "salida"
SAL.mkdir(parents=True, exist_ok=True)

# Paleta categorica (orden fijo) y tintas de texto.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 7,
})

MD = []  # secciones de tablas.md


def tabla(df, nombre, titulo, fmt=None):
    df.to_csv(SAL / f"{nombre}.csv", index=False)
    d = df.copy()
    for col, f in (fmt or {}).items():
        if col in d:
            d[col] = d[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
    MD.append(f"### {titulo}\n\n" + d.to_markdown(index=False) + "\n")


def guardar(fig, nombre):
    fig.savefig(SAL / f"{nombre}.png")
    plt.close(fig)
    print("  figura:", nombre + ".png")


def gflops(N, t):
    return 2 * N ** 3 / 3 / t / 1e9


# ----------------------------------------------------------------------------- (a)
def parte_a():
    f = RAIZ / "resultados_lu_serial.txt"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t")
    g = df.groupby(["N", "b"])["tiempo_s"]
    r = pd.DataFrame({"mediana_s": g.median(), "min_s": g.min(), "max_s": g.max(),
                      "reps": g.count()}).reset_index().sort_values("b", ascending=False)
    r["GFLOP/s"] = gflops(r["N"], r["mediana_s"])
    b_a = int(r.loc[r["mediana_s"].idxmin(), "b"])
    tabla(r, "a_serial", f"(a) LU serial local, N={r['N'].iloc[0]}, 1 thread (mediana de {r['reps'].max()} reps). b elegido = {b_a}",
          {"mediana_s": "{:.3f}", "min_s": "{:.3f}", "max_s": "{:.3f}", "GFLOP/s": "{:.1f}"})

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.log2(r["b"])
    ax.fill_between(x, r["min_s"], r["max_s"], color=C[0], alpha=0.15, lw=0, label="rango (min-max)")
    ax.plot(x, r["mediana_s"], "o-", color=C[0], label="mediana")
    ax.plot(np.log2(b_a), r["mediana_s"].min(), "o", ms=13, mfc="none", mec=INK, mew=1.5)
    ax.annotate(f"b = {b_a}", (np.log2(b_a), r["mediana_s"].min()), xytext=(0, -22),
                textcoords="offset points", ha="center", color=INK)
    ax.set_xticks(x, [f"$2^{{{int(v)}}}$" for v in x])
    ax.set_xlabel("Tamaño de bloque b")
    ax.set_ylabel("Tiempo de factorización [s]")
    ax.set_title(f"(a) LU serial, N = {r['N'].iloc[0]}, 1 thread")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    guardar(fig, "a_serial_tiempo_vs_b")
    return b_a


# ----------------------------------------------------------------------------- datos cluster
def cargar_cluster():
    archivos = [f for f in glob.glob(str(RAIZ / "resultados_cluster" / "*.csv"))
                if "check_" not in f]
    if not archivos:
        return None, None
    partes = []
    for f in archivos:
        d = pd.read_csv(f)
        d["archivo"] = Path(f).name
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df["T"] = df["T"].fillna(1).astype(int)
    df["grid"] = df["pr"].astype(str) + "x" + df["pc"].astype(str)
    df["CPUs"] = df["procesos"] * df["T"]
    claves = ["archivo", "fecha", "rep", "parte", "config", "N", "b", "pr", "pc", "grid",
              "procesos", "T", "CPUs", "nodos"]
    run = df.groupby(claves, dropna=False).agg(
        tiempo_s=("tiempo_global_s", "max"),
        comp_prom=("computo_s", "mean"), comp_max=("computo_s", "max"),
        comm_prom=("comunicacion_s", "mean"), comm_max=("comunicacion_s", "max"),
        diag=("diag_s", "mean"), panel=("panel_s", "mean"), update=("update_s", "mean"),
        bcast_diag=("bcast_diag_s", "mean"), bcast_panel=("bcast_panel_s", "mean"),
        bytes_bcast=("bytes_bcast", "mean"),
    ).reset_index()
    return df, run


def resumir(run, por):
    cols = ["tiempo_s", "comp_prom", "comp_max", "comm_prom", "comm_max",
            "diag", "panel", "update", "bcast_diag", "bcast_panel", "bytes_bcast"]
    g = run.groupby(por)
    r = g[cols].median().reset_index()
    r["reps"] = g.size().values
    r["t_min"] = g["tiempo_s"].min().values
    r["t_max"] = g["tiempo_s"].max().values
    return r


# ----------------------------------------------------------------------------- (c)
def parte_c(run, b_a):
    c = run[run["parte"] == "c"]
    if c.empty:
        return None
    r = resumir(c, ["N", "b"]).sort_values("b", ascending=False)
    r["GFLOP/s"] = gflops(r["N"], r["tiempo_s"])
    b_c = int(r.loc[r["tiempo_s"].idxmin(), "b"])
    t = r[["b", "tiempo_s", "t_min", "t_max", "comp_prom", "comm_prom", "comp_max", "comm_max", "GFLOP/s", "reps"]]
    tabla(t, "c_variacion_b", f"(c) Cluster, N={r['N'].iloc[0]}, grid 2x2, T=2 (mediana). b elegido = {b_c} (en (a): {b_a})",
          {k: "{:.3f}" for k in ["tiempo_s", "t_min", "t_max", "comp_prom", "comm_prom", "comp_max", "comm_max"]} | {"GFLOP/s": "{:.1f}"})

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.log2(r["b"])
    ax.fill_between(x, r["t_min"], r["t_max"], color=C[0], alpha=0.15, lw=0)
    ax.plot(x, r["tiempo_s"], "o-", color=C[0], label="tiempo total (máx. entre ranks)")
    ax.plot(x, r["comp_prom"], "s--", color=C[1], label="cómputo (prom. por rank)")
    ax.plot(x, r["comm_prom"], "^--", color=C[2], label="comunicación (prom. por rank)")
    ax.set_xticks(x, [f"$2^{{{int(v)}}}$" for v in x])
    ax.set_xlabel("Tamaño de bloque b")
    ax.set_ylabel("Tiempo [s]")
    ax.set_title(f"(c) N = {r['N'].iloc[0]}, grid 2×2, T = 2: tiempo vs b")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    guardar(fig, "c_tiempo_vs_b")

    # Desglose para (g): que componente domina para cada b.
    comps = [("update", "actualización (GEMM)"), ("panel", "paneles (trsm)"), ("diag", "bloque diagonal (getrf)"),
             ("bcast_panel", "Bcast paneles"), ("bcast_diag", "Bcast diagonal")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    base = np.zeros(len(r))
    labels = [str(int(b)) for b in r["b"]]
    for i, (k, nom) in enumerate(comps):
        ax.bar(labels, r[k], bottom=base, color=C[i], label=nom, edgecolor="white", linewidth=1)
        base += r[k].values
    ax.set_xlabel("Tamaño de bloque b")
    ax.set_ylabel("Tiempo promedio por rank [s]")
    ax.set_title("(c)/(g) Desglose del tiempo por componente, grid 2×2, T = 2")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_ylim(0, base.max() * 1.35)
    guardar(fig, "g_desglose_componentes_vs_b")
    tabla(r[["b", "diag", "panel", "update", "bcast_diag", "bcast_panel", "bytes_bcast"]],
          "g_desglose_c", "(g) Desglose medido por rank (promedio), grid 2x2, T=2",
          {k: "{:.3f}" for k in ["diag", "panel", "update", "bcast_diag", "bcast_panel"]} | {"bytes_bcast": "{:.3e}"})
    return b_c


# ----------------------------------------------------------------------------- (d)
def parte_d(run):
    d = run[(run["parte"] == "d") & (run["nodos"] == 1)]
    if d.empty:
        return
    r = resumir(d, ["config", "grid", "pr", "pc", "procesos", "T", "CPUs", "N", "b"])
    base = r[(r["config"].astype(str) == "0") & (r["T"] == 1)].set_index("b")["tiempo_s"]
    r["speedup"] = r.apply(lambda x: base.get(x["b"], np.nan) / x["tiempo_s"], axis=1)
    r["eficiencia"] = r["speedup"] / r["CPUs"]
    r["GFLOP/s"] = gflops(r["N"], r["tiempo_s"])
    r = r.sort_values(["b", "config", "grid", "T"])
    tabla(r[["b", "config", "grid", "T", "CPUs", "tiempo_s", "comp_prom", "comm_prom", "speedup", "eficiencia", "GFLOP/s", "reps"]],
          "d_speedup", "(d) Configuraciones 0-3, N=8192 (mediana). S = t(1x1,T=1)/t ; E = S/CPUs",
          {"tiempo_s": "{:.3f}", "comp_prom": "{:.3f}", "comm_prom": "{:.3f}", "speedup": "{:.2f}", "eficiencia": "{:.2f}", "GFLOP/s": "{:.1f}"})

    bs = sorted(r["b"].unique())
    fig, axs = plt.subplots(1, len(bs), figsize=(6.2 * len(bs), 4.6), squeeze=False, sharey=True)
    for ax, b in zip(axs[0], bs):
        rb = r[r["b"] == b]
        for i, (g, sub) in enumerate(rb.groupby("grid", sort=False)):
            sub = sub.sort_values("CPUs")
            ax.plot(sub["CPUs"], sub["speedup"], "o-", color=C[i], label=f"grid {g} (T = {', '.join(map(str, sub['T']))})")
        ax.plot([1, 8], [1, 8], ":", color=INK2, lw=1.5, label="ideal")
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"]); ax.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        ax.set_xlabel("CPUs totales (p × T)")
        ax.set_title(f"b = {b}")
        ax.legend(frameon=False, fontsize=9)
    axs[0][0].set_ylabel("Speedup vs. 1×1, T = 1")
    fig.suptitle(f"(d) Speedup, configuraciones 0-3, N = {int(r['N'].iloc[0])}", y=1.02)
    guardar(fig, "d_speedup")

    # Eficiencia en barras, una por (config, grid, T).
    fig, ax = plt.subplots(figsize=(10, 4.2))
    etiquetas = rb_lab = None
    w = 0.8 / len(bs)
    for j, b in enumerate(bs):
        rb = r[r["b"] == b].reset_index(drop=True)
        rb_lab = [f"{g}\nT={t}" for g, t in zip(rb["grid"], rb["T"])]
        xs = np.arange(len(rb)) + (j - (len(bs) - 1) / 2) * w
        ax.bar(xs, rb["eficiencia"], w * 0.92, color=C[j], label=f"b = {b}")
    ax.set_xticks(np.arange(len(rb_lab)), rb_lab, fontsize=9)
    ax.axhline(1, color=INK2, ls=":", lw=1.2)
    ax.set_ylabel("Eficiencia S / CPUs")
    ax.set_title("(d) Eficiencia por configuración")
    ax.legend(frameon=False)
    guardar(fig, "d_eficiencia")


# ----------------------------------------------------------------------------- (e)
def parte_e(run):
    sel = run[((run["config"].astype(str) == "3") & (run["grid"] == "2x4")) |
              (run["config"].astype(str).isin(["4", "5"]))]
    if sel.empty:
        return
    r = resumir(sel, ["config", "grid", "procesos", "T", "CPUs", "nodos", "b"]).sort_values(["b", "config"])
    r["ancho_banda_ef_GBs"] = r["bytes_bcast"] / (r["bcast_diag"] + r["bcast_panel"]) / 1e9
    tabla(r[["b", "config", "grid", "nodos", "CPUs", "tiempo_s", "comp_prom", "comm_prom", "comp_max", "comm_max",
             "bcast_diag", "bcast_panel", "ancho_banda_ef_GBs", "reps"]],
          "e_1nodo_vs_2nodos", "(e) Config 3 (2x4, 1 nodo) vs 4 (2x4, 2 nodos) vs 5 (4x4, 2 nodos). Ancho de banda efectivo = bytes recibidos/enviados en Bcast / tiempo en Bcast (incluye espera)",
          {k: "{:.3f}" for k in ["tiempo_s", "comp_prom", "comm_prom", "comp_max", "comm_max", "bcast_diag", "bcast_panel"]} | {"ancho_banda_ef_GBs": "{:.2f}"})

    bs = sorted(r["b"].unique())
    fig, axs = plt.subplots(1, len(bs), figsize=(5.8 * len(bs), 4.4), squeeze=False, sharey=True)
    for ax, b in zip(axs[0], bs):
        rb = r[r["b"] == b]
        lab = [f"cfg {c}\n{g}, {n} nodo{'s' if n > 1 else ''}" for c, g, n in zip(rb["config"], rb["grid"], rb["nodos"])]
        ax.bar(lab, rb["comp_prom"], color=C[0], label="cómputo", edgecolor="white")
        ax.bar(lab, rb["comm_prom"], bottom=rb["comp_prom"], color=C[1], label="comunicación", edgecolor="white")
        for x, (cp, cm) in enumerate(zip(rb["comp_prom"], rb["comm_prom"])):
            ax.text(x, cp + cm, f"{cp + cm:.2f} s", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_title(f"b = {b}")
    axs[0][0].set_ylabel("Tiempo promedio por rank [s]")
    axs[0][0].legend(frameon=False)
    fig.suptitle(f"(e) Cómputo vs. comunicación: 1 nodo vs. 2 nodos, N = {int(sel['N'].iloc[0])}", y=1.02)
    guardar(fig, "e_1nodo_vs_2nodos")


# ----------------------------------------------------------------------------- (f)
def parte_f(df):
    N = int(df["N"].max())
    f = df[(df["grid"] == "2x2") & (df["parte"].isin(["c", "f"])) & (df["b"].isin([N // 2, N // 16]))]
    if f.empty:
        return
    r = f.groupby(["T", "b", "rank", "fila", "columna"])[["computo_s", "comunicacion_s", "total_s", "update_s", "panel_s", "diag_s"]].median().reset_index()
    r["proceso"] = "(" + r["fila"].astype(str) + "," + r["columna"].astype(str) + ")"
    tabla(r[["T", "b", "rank", "proceso", "computo_s", "comunicacion_s", "total_s", "diag_s", "panel_s", "update_s"]],
          "f_por_rank", "(f) Grid 2x2, N=8192: b=N/2 vs b=N/16, por proceso (mediana)",
          {k: "{:.3f}" for k in ["computo_s", "comunicacion_s", "total_s", "diag_s", "panel_s", "update_s"]})
    Ts = sorted(r["T"].unique())
    fig, axs = plt.subplots(1, len(Ts), figsize=(6.5 * len(Ts), 4.4), squeeze=False, sharey=True)
    for ax, T in zip(axs[0], Ts):
        rt = r[r["T"] == T]
        labs, comp, comm = [], [], []
        for b in (N // 2, N // 16):
            for _, x in rt[rt["b"] == b].sort_values("rank").iterrows():
                labs.append(f"{x['proceso']}\nb={b}"); comp.append(x["computo_s"]); comm.append(x["comunicacion_s"])
        xs = np.arange(len(labs)) + np.repeat([0, 0.6], len(labs) // 2 if len(labs) >= 2 else 1)[:len(labs)]
        ax.bar(xs, comp, 0.8, color=C[0], label="cómputo", edgecolor="white")
        ax.bar(xs, comm, 0.8, bottom=comp, color=C[1], label="comunicación", edgecolor="white")
        ax.set_xticks(xs, labs, fontsize=8.5)
        ax.set_title(f"T = {T}")
    axs[0][0].set_ylabel("Tiempo [s]")
    axs[0][0].legend(frameon=False)
    fig.suptitle(f"(f) Grid 2×2, N = {N}: b = N/2 = {N//2} vs. b = N/16 = {N//16}, por proceso", y=1.02)
    guardar(fig, "f_por_proceso")


# ----------------------------------------------------------------------------- (b) correctitud
def correctitud():
    filas = []
    for log in glob.glob(str(RAIZ / "logs" / "lu_prueba_entorno_*.out")):
        cfg = None
        for line in open(log, errors="ignore"):
            m = re.match(r"--- CHECK p=(\d+) grid=(\S+) N=(\d+) b=(\d+)", line)
            if m:
                cfg = m.groups()
            m = re.search(r"error_relativo_vs_serial=(\S+) residuo_relativo=(\S+)", line)
            if m and cfg:
                filas.append(dict(p=int(cfg[0]), grid=cfg[1], N=int(cfg[2]), b=int(cfg[3]),
                                  error_vs_serial=m.group(1), residuo=m.group(2)))
    if filas:
        tabla(pd.DataFrame(filas), "b_correctitud", "(b) Correctitud en el cluster: ||LU_dist - LU_serial||/||LU_serial|| y ||A-LU||/||A||")


def main():
    print("Analizando...")
    b_a = parte_a()
    df, run = cargar_cluster()
    correctitud()
    if run is not None:
        run.to_csv(SAL / "todas_las_corridas.csv", index=False)
        b_c = parte_c(run, b_a)
        parte_d(run)
        parte_e(run)
        parte_f(df)
        print(f"b_a = {b_a}, b_c = {b_c}")
    else:
        print("No hay resultados del cluster todavia (resultados_cluster/*.csv).")
    (SAL / "tablas.md").write_text("# Tablas Tarea 3\n\n" + "\n".join(MD), encoding="utf-8")
    print("Tablas en", SAL / "tablas.md")


if __name__ == "__main__":
    main()
