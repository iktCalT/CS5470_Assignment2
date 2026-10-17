from datetime import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze.py <path_to_result_1_folder> <path_to_result_2_folder>")
        sys.exit(1)

    path_1 = sys.argv[1]
    path_2 = sys.argv[2]

    # --- 3.1 --- 
    ttfts, tpots = read_client(os.path.join(path_1, "client.txt"))

    # Not same as report, because the prompts "Appending: 842, 1" 
    # and "Appending: 77, 1" are discarded in the report
    print(f"Shape of TTFTs {ttfts.shape}, ttft_req10_s = {ttfts[10]}, ttft_req40_s = {ttfts[40]}")

    # 3.1.1 TTFTs
    fig_ttft, ax_ttft = plt.subplots()
    plot(ttfts, "ttfts", ax_ttft)
    ax_ttft.set_title("TTFTs")
    # ax_ttft.legend()
    fig_ttft.savefig(os.path.join(path_1, "ttft.png"), dpi=300)

    # 3.1.2 TPOTs
    fig_tpot, ax_tpot = plt.subplots()
    plot(tpots, "tpots", ax_tpot)
    ax_tpot.set_title("TPOTs")
    # ax_tpot.legend()
    fig_tpot.savefig(os.path.join(path_1, "tpot.png"), dpi=300)

    # 3.1.3 GPU KV cache usage
    time, kv_usages, _, _ = read_server(os.path.join(path_1, "vllm_server.log"))

    fig_kv_usage, ax_kv_usage = plt.subplots()
    ax_kv_usage.plot(time, kv_usages)
    ax_kv_usage.set_xlabel("Time (s)")
    ax_kv_usage.set_ylabel("GPU KV cache usage (%)")
    ax_kv_usage.set_title("GPU KV Cache Usage (%)")
    # ax_kv_usage.legend()
    fig_kv_usage.savefig(os.path.join(path_1, "gpu_kv.png"), dpi=300)

    # --- 3.2 ---
    ttfts_0, tpots_0 = read_client(os.path.join(path_2, "p0_r1.txt"))
    ttfts_10, tpots_10 = read_client(os.path.join(path_2, "p10_r1.txt"))

    # 3.2.1 TTFTs
    fig_ttft_comp, ax_ttft_comp = plt.subplots() # comp: compare
    plot(ttfts_0, "TTFT", ax_ttft_comp, "FCFS (interval 0)")
    plot(ttfts_10, "TTFT", ax_ttft_comp, "CFS (interval 10)")
    ax_ttft_comp.set_title("TTFT: FCFS vs. CFS (interval 10)")
    ax_ttft_comp.legend()
    fig_ttft_comp.savefig(os.path.join(path_2, "ttft.png"), dpi=300)

    # 3.2.2 TPOTs
    fig_tpot_comp, ax_tpot_comp = plt.subplots()
    plot(tpots_0, "TPOT", ax_tpot_comp, "FCFS (interval 0)")
    plot(tpots_10, "TPOT", ax_tpot_comp, "CFS (interval 10)")
    ax_tpot_comp.set_title("TPOT: FCFS vs. CFS (interval 10)")
    ax_tpot_comp.legend()
    fig_tpot_comp.savefig(os.path.join(path_2, "tpot.png"), dpi=300)

    # 3.2.{3, 4, 5}
    t_0, gpu_0, cpu_0, wait_0 = read_server(os.path.join(path_2, "server_p0.log"))
    t_10, gpu_10, cpu_10, wait_10 = read_server(os.path.join(path_2, "server_p10.log"))

    # 3.2.3 GPU KV cache usage
    fig_gpu_comp, ax_gpu_comp = plt.subplots()
    ax_gpu_comp.plot(t_0, gpu_0, label="FCFS (interval 0)")
    ax_gpu_comp.plot(t_10, gpu_10, label="CFS (interval 10)")
    ax_gpu_comp.set_xlabel("Time (s)")
    ax_gpu_comp.set_ylabel("GPU KV cache usage (%)")
    ax_gpu_comp.set_title("GPU KV Cache Usage: FCFS vs. CFS (interval 10)")
    ax_gpu_comp.legend()
    fig_gpu_comp.savefig(os.path.join(path_2, "gpu_kv.png"), dpi=300)

    # 3.2.4 CPU KV cache usage
    fig_cpu_comp, ax_cpu_comp = plt.subplots()
    ax_cpu_comp.plot(t_0, cpu_0, label="FCFS (interval 0)")
    ax_cpu_comp.plot(t_10, cpu_10, label="CFS (interval 10)")
    ax_cpu_comp.set_xlabel("Time (s)")
    ax_cpu_comp.set_ylabel("CPU KV cache usage (%)")
    ax_cpu_comp.set_title("CPU KV Cache Usage: FCFS vs. CFS (interval 10)")
    ax_cpu_comp.legend()
    fig_cpu_comp.savefig(os.path.join(path_2, "cpu_kv.png"), dpi=300)

    # 3.2.5 number of waiting requests
    fig_wait_comp, ax_wait_comp = plt.subplots()
    ax_wait_comp.plot(t_0, wait_0, label="FCFS (interval 0)")
    ax_wait_comp.plot(t_10, wait_10, label="CFS (interval 10)")
    ax_wait_comp.set_xlabel("Time (s)")
    ax_wait_comp.set_ylabel("# waiting requests")
    ax_wait_comp.set_title("Number of Waiting Requests: FCFS vs. CFS (interval 10)")
    ax_wait_comp.legend()
    fig_wait_comp.savefig(os.path.join(path_2, "waiting.png"), dpi=300)

    # 3.2.6 all intervals
    intervals = [0, 5, 10, 20, 40]
    labels = ["FCFS", "5", "10", "20", "40"]
    p99_ttfts = []
    p99_tpots = []

    for p, label in zip(intervals, labels):
        c_ttfts, c_tpots = read_client(os.path.join(path_2, f"p{p}_r1.txt"))
        p99_ttft = np.nanpercentile(c_ttfts, 99) * 1000
        p99_tpot = np.nanpercentile(c_tpots, 99) * 1000
        p99_ttfts.append(p99_ttft)
        p99_tpots.append(p99_tpot)
        print(f"Interval {label}: P99 TTFT = {p99_ttft:.2f} ms, P99 TPOT = {p99_tpot:.2f} ms")

    fig_pareto, ax_pareto = plt.subplots()
    ax_pareto.scatter(p99_tpots, p99_ttfts)
    for label, x, y in zip(labels, p99_tpots, p99_ttfts):
        ax_pareto.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    ax_pareto.set_xlabel("P99 TPOT (ms)")
    ax_pareto.set_ylabel("P99 TTFT (ms)")
    ax_pareto.set_title("P99 TTFT vs. P99 TPOT: All Intervals")
    fig_pareto.savefig(os.path.join(path_2, "pareto.png"), dpi=300)


def read_client(filename: str) -> tuple[np.ndarray, np.ndarray]:
    ttfts = []
    tpots = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("TTFT:"):
                parts = line.split()

                ttft = float(parts[1])
                tpot = float(parts[3])

                ttfts.append(ttft)
                tpots.append(tpot)

    ttfts = np.array(ttfts)
    tpots = np.array(tpots)
    
    print(filename)
    print(f"p99 TTFTs: {np.nanpercentile(ttfts, 99)} s")
    print(f"p99 TPOTs: {np.nanpercentile(tpots, 99) * 1000} ms")
    
    return ttfts, tpots


def read_server(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = []
    kv_usages = []
    cpu_usages = []
    pending = []
    swapped = []

    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
        r".*Swapped:\s*(\d+)\s*reqs"
        r".*Pending:\s*(\d+)\s*reqs"
        r".*GPU KV cache usage:\s*([\d.]+)%"
        r".*CPU KV cache usage:\s*([\d.]+)%"
    )

    start_pending = None
    end_pending = None
    with open(filename) as f:
        for line in f:
            if match := pattern.search(line):
                t = datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                )
                swap = int(match.group(2))
                pend = int(match.group(3))
                gpu_usage = float(match.group(4))
                cpu_usage = float(match.group(5))

                time.append(t)
                pending.append(pend)
                swapped.append(swap)
                kv_usages.append(gpu_usage)
                cpu_usages.append(cpu_usage)
                
                if start_pending is None and pend > 0:
                    start_pending = t
                if start_pending is not None and pend == 0 and end_pending is None:
                    end_pending = t

    relative_time = np.array([
        (t - time[0]).total_seconds() for t in time
    ])
    
    print(f"{filename} \
        Maximum GPU KV cache usage = {max(kv_usages)}\
        Pending waiting time = {(end_pending - start_pending).total_seconds()} s\
        Maximum pending count = {max(pending)}\
        Maximum swapped count = {max(swapped)}\
        Maximum CPU KV cache usage = {max(cpu_usages)}")

    return (
        relative_time,
        np.array(kv_usages),
        np.array(cpu_usages),
        np.array(pending),
    )


def plot(y: np.ndarray, label: str, ax, legend_label: str = None):
    x = range(len(y))

    ax.plot(x, 1000 * y, label=legend_label)

    ax.set_xlabel("prompt index")
    ax.set_ylabel("{} (ms)".format(label))


if __name__ == "__main__":
    main()