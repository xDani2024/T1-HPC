#!/usr/bin/env python3
import subprocess
import csv
import re
import numpy as np

# Configuration parameters according to task specification
N = 2048
N_VALUES = [2048, 1024, 512, 256, 128, 64, 32]
P_VALUES = [1, 2, 4, 8]
REPETITIONS = 3
EXECUTABLE = "./algoritmo2"
OUTPUT_CSV = "algoritmo2_experiments.csv"

def parse_output(output_str):
    """
    Parses output format:
    N=2048 n=256 threads=4 tiempo=123.45 ms checksum=...
    """
    time_match = re.search(r"tiempo=([\d\.]+)\s*ms", output_str)
    checksum_match = re.search(r"checksum=([\d\.\-]+)", output_str)
    
    if time_match and checksum_match:
        return float(time_match.group(1)), checksum_match.group(1)
    else:
        raise ValueError(f"Could not parse output: {output_str}")

def main():
    print(f"=== Starting Experiments for Algorithm 2 (N={N}) ===")
    results = []
    
    # Map to store baseline T(1) averages for each n to calculate speedup
    baseline_times = {}

    for n in N_VALUES:
        for p in P_VALUES:
            print(f"Running: n={n}, p={p} ({REPETITIONS} runs)...", end="", flush=True)
            times = []
            checksums = []

            for run in range(REPETITIONS):
                try:
                    cmd = [EXECUTABLE, str(N), str(n), str(p)]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    t_ms, chk = parse_output(res.stdout)
                    times.append(t_ms)
                    checksums.append(chk)
                except Exception as e:
                    print(f"\nError running command {' '.join(cmd)}: {e}")
                    return

            avg_time = np.mean(times)
            std_time = np.std(times)
            chk_sample = checksums[0]

            # Cache baseline execution time for p=1
            if p == 1:
                baseline_times[n] = avg_time

            speedup = baseline_times[n] / avg_time if n in baseline_times else 1.0
            efficiency = speedup / p

            results.append({
                "N": N,
                "n": n,
                "p": p,
                "run_1_ms": round(times[0], 2),
                "run_2_ms": round(times[1], 2),
                "run_3_ms": round(times[2], 2),
                "avg_time_ms": round(avg_time, 2),
                "std_dev_ms": round(std_time, 2),
                "speedup": round(speedup, 2),
                "efficiency": round(efficiency, 2),
                "checksum": chk_sample
            })

            print(f" Done. Avg: {avg_time:.2f} ms | Checksum: {chk_sample}")

    # Write output to CSV
    fieldnames = [
        "N", "n", "p", 
        "run_1_ms", "run_2_ms", "run_3_ms", 
        "avg_time_ms", "std_dev_ms", 
        "speedup", "efficiency", "checksum"
    ]

    with open(OUTPUT_CSV, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Experiments completed successfully! Results exported to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()