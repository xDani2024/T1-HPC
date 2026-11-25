#!/usr/bin/env python3
"""Tarea 4 (d): mediciones locales de NumPy y JAX en CPU.
Las funciones matemáticas provienen del notebook completado por el grupo.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

def medir_worker(region, W, metodo, reps):
    import numpy as np
    import jax
    import jax.numpy as jnp

    MAX_ITER = 100
    TOL = 1e-6
    ROOTS = np.array([1.0, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)], dtype=np.complex64)

    REGIONS = {
        "clasica": (-2.0, 2.0, -2.0, 2.0),
        "frontera_central": (-0.3, 0.3, -0.3, 0.3),
        "frontera_lateral": (0.3, 0.9, 0.3, 0.9),
    }


    def grid(region, W):
        x_min, x_max, y_min, y_max = REGIONS[region]
        x = np.linspace(x_min, x_max, W, dtype=np.float32)
        y = np.linspace(y_min, y_max, W, dtype=np.float32)
        return (x[None, :] + 1j * y[:, None]).astype(np.complex64)


    def newton_numpy(z0):
        z = z0.copy()
        count = np.zeros(z.shape, dtype=np.int32)
        active = np.ones(z.shape, dtype=bool)
        for _ in range(MAX_ITER):
            z_new = z[active] - (z[active] ** 3 - 1) / (3 * z[active] ** 2)
            converged = np.abs(z_new - z[active]) < TOL
            z[active] = z_new
            count[active] += 1
            active[active] = ~converged
            if not active.any():
                break
        root_idx = np.full(z.shape, -1, dtype=np.int32)
        for i, r in enumerate(ROOTS):
            root_idx[np.abs(z - r) < 1e-3] = i
        return root_idx, count

    def pixel(z0):
      def cond_fn(state):
        z, count, active = state
        return (count < MAX_ITER) & active

      def body_fn(state):
        z, count, active = state

        z_new = z - (z**3 - 1) / (3 * z**2)
        converged = jnp.abs(z_new - z) < TOL

        return z_new, count + 1, ~converged

      z, count, _ = jax.lax.while_loop(
          cond_fn,
          body_fn,
          (z0, jnp.int32(0), jnp.bool_(True))
      )

      root_idx = jnp.int32(-1)
      for i, r in enumerate(ROOTS):
          root_idx = jnp.where(
              jnp.abs(z - jnp.asarray(r)) < 1e-3,
              jnp.int32(i),
              root_idx
          )

      return root_idx, count

    @jax.jit
    def newton_jax(z0):
      return jax.vmap(jax.vmap(pixel))(z0)

    z_np = grid(region, W)
    if metodo == "NumPy":
        def ejecutar():
            newton_numpy(z_np)
    else:
        if any(d.platform != "cpu" for d in jax.devices()):
            raise RuntimeError("JAX no está ejecutándose en CPU")
        z_jax = jax.device_put(z_np, jax.devices("cpu")[0])
        jax.block_until_ready(z_jax)
        def ejecutar():
            jax.block_until_ready(newton_jax(z_jax))
        ejecutar()  # warm-up y compilación fuera del tiempo medido

    tiempos = []
    for _ in range(reps):
        inicio = time.perf_counter()
        ejecutar()
        tiempos.append(time.perf_counter() - inicio)
    return tiempos


def hardware(path):
    lines = [f"Sistema: {platform.platform()}", f"Python: {platform.python_version()}",
             f"CPU lógicos visibles: {os.cpu_count()}",
             f"CPU lógicos disponibles (afinidad): {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}",
             "Dispositivo JAX solicitado: CPU (JAX_PLATFORMS=cpu)"]
    for pkg in ("numpy", "jax", "jaxlib"):
        lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
    for cmd in (["lscpu"], ["free", "-h"]):
        try:
            lines.append(subprocess.check_output(cmd, text=True))
        except (OSError, subprocess.CalledProcessError):
            pass
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024,2048,4096,8192])
    parser.add_argument("--regions", nargs="+", choices=["clasica", "frontera_central", "frontera_lateral"], default=["clasica", "frontera_central", "frontera_lateral"])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("resultados_local.csv"))
    parser.add_argument("--worker", nargs=3, metavar=("REGION", "W", "METODO"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reps < 1 or any(w < 2 for w in args.sizes):
        parser.error("Se requiere reps >= 1 y W >= 2")
    if args.worker:
        region, W, metodo = args.worker
        print(json.dumps(medir_worker(region, int(W), metodo, args.reps)))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hw = args.output.with_name(args.output.stem + "_hardware.txt")
    hardware(hw)
    print(f"Hardware: {hw}", flush=True)
    campos = ["Region", "W", "NumPy (s)", "JAX (s)", "NumPy repeticiones (s)", "JAX repeticiones (s)", "Estado"]
    errores = 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        f.flush()
        for region in args.regions:
            for W in args.sizes:
                row = {"Region": region, "W": W, "Estado": "OK"}
                for metodo in ("NumPy", "JAX"):
                    print(f"{region}, W={W}, {metodo} ...", flush=True)
                    # Un proceso independiente por medición/configuración evita acumular memoria y cachés de JAX.
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", region, str(W), metodo, "--reps", str(args.reps)]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        errores += 1
                        row["Estado"] = ("" if row["Estado"] == "OK" else row["Estado"] + "; ") + f"{metodo}: salida {result.returncode}"
                        print(f"  ERROR ({result.returncode}); revisar RAM o mensaje anterior.", flush=True)
                    else:
                        tiempos = json.loads(result.stdout.strip().splitlines()[-1])
                        mediana = statistics.median(tiempos)
                        row[f"{metodo} (s)"] = f"{mediana:.9f}"
                        row[f"{metodo} repeticiones (s)"] = ";".join(f"{t:.9f}" for t in tiempos)
                        print(f"  {mediana:.6f} s", flush=True)
                    # Guardar resultados incluso si una medición posterior falla.
                writer.writerow(row)
                f.flush()
    print(f"Resultados: {args.output.resolve()}", flush=True)
    if errores:
        print(f"Mediciones fallidas: {errores}. Las celdas correspondientes quedan vacías.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
