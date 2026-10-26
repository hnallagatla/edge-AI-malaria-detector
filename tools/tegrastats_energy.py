import re
import argparse
from datetime import datetime

TS_RE = re.compile(r'^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})')
def rail_re(rail_name: str):
    return re.compile(rf'\b{re.escape(rail_name)}\s+(\d+)mW/(\d+)mW')

def parse_log(path, rail):
    r = rail_re(rail)
    samples = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            mts = TS_RE.search(line)
            mrail = r.search(line)
            if not (mts and mrail):
                continue
            ts = datetime.strptime(mts.group(1), "%m-%d-%Y %H:%M:%S")
            cur_mw = int(mrail.group(1))
            avg_mw = int(mrail.group(2))
            samples.append((ts, cur_mw, avg_mw))
    samples.sort(key=lambda x: x[0])
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="tegrastats log file (text)")
    ap.add_argument("--rail", default="VDD_IN", help="Power rail to use (default: VDD_IN)")
    ap.add_argument("--assume_dt", type=float, default=None,
                    help="If set, use this fixed dt (seconds) instead of timestamp differences")
    args = ap.parse_args()

    samples = parse_log(args.file, args.rail)
    if len(samples) < 2:
        raise SystemExit("Not enough samples found (need >= 2 lines with timestamps and the rail).")

    dts = []
    powers_w = []
    for i in range(len(samples) - 1):
        ts, cur_mw, _ = samples[i]
        ts2, _, _ = samples[i + 1]
        dt = (ts2 - ts).total_seconds()
        if args.assume_dt is not None:
            dt = args.assume_dt
        if dt <= 0:
            continue
        dts.append(dt)
        powers_w.append(cur_mw / 1000.0)

    if not dts:
        raise SystemExit("No valid dt intervals found.")

    total_time_s = sum(dts)
    total_energy_j = sum(p * dt for p, dt in zip(powers_w, dts))
    avg_power_w = total_energy_j / total_time_s
    total_energy_wh = total_energy_j / 3600.0

    print(f"Rail: {args.rail}")
    print(f"Samples used: {len(dts)} intervals")
    print(f"Total time (s): {total_time_s:.3f}")
    print(f"Average power (W): {avg_power_w:.6f}")
    print(f"Total energy (J): {total_energy_j:.3f}")
    print(f"Total energy (Wh): {total_energy_wh:.6f}")

if __name__ == '__main__':
    main()