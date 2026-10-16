from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np

def main():
    ttfts, tpots = read_client("./results_3.1/client.txt")
    
    # Not same as report, because the prompts "Appending: 842, 1" 
    # and "Appending: 77, 1" are discarded in the report
    print(f"Shape of TTFTs {ttfts.shape}, ttft_req10_s = {ttfts[10]}, ttft_req40_s = {ttfts[40]}")
    print(f"Mean TTFTs: {ttfts.mean() * 1000} ms; \
          p99 TTFTs: {np.nanpercentile(ttfts, 99) * 1000} ms")
    print(f"Mean TPOTs: {tpots.mean() * 1000} ms; \
          p99 TPOTs: {np.nanpercentile(tpots, 99) * 1000} ms")

    fig_ttft, ax_ttft = plt.subplots()
    fig_tpot, ax_tpot = plt.subplots()
    
    plot(ttfts, "ttfts", ax_ttft)
    plot(tpots, "tpots", ax_tpot)
    
    ax_ttft.set_title("TTFTs")
    fig_ttft.savefig("./imgs_3.1/ttfts.png", dpi=300)
    
    ax_tpot.set_title("TPOTs")
    fig_tpot.savefig("./imgs_3.1/tpots.png", dpi=300)
    
    time, kv_usages = read_server("./results_3.1/vllm_server.log")
    print(f"Maximum KV cache usage = {kv_usages.max()}")
    
    fig_kv_usage, ax_kv_usage = plt.subplots()
    ax_kv_usage.plot(time, kv_usages)

    ax_kv_usage.set_xlabel("Time (s)")
    ax_kv_usage.set_ylabel("GPU KV cache usage (%)")
    fig_kv_usage.savefig("./imgs_3.1/kv_usage.png", dpi=300)

    
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
    
    return np.array(ttfts), np.array(tpots)

def read_server(filename: str) -> tuple[np.ndarray, np.ndarray]:
    time = []
    kv_usages = []

    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
        r".*GPU KV cache usage:\s*([\d.]+)%"
    )

    with open(filename) as f:
        for line in f:
            if match := pattern.search(line):
                t = datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                )
                usage = float(match.group(2))

                time.append(t)
                kv_usages.append(usage)

    relative_time = np.array([
        (t - time[0]).total_seconds() for t in time
    ])

    return relative_time, np.array(kv_usages)

    
def plot(y: np.ndarray, label: str, ax):
    x = range(len(y))

    ax.plot(x, 1000 * y)

    ax.set_xlabel("prompt index")
    ax.set_ylabel("{} (ms)".format(label))
    
if __name__ == "__main__":
    main()