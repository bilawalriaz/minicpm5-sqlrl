"""Watch a training run and stop it on clear failure. Runs beside train.py.

    python scripts/monitor.py --run NAME --pid PID

Every minute it appends a summary to runs/NAME/monitor.jsonl. It sends SIGTERM (the
trainer saves an interrupted checkpoint) when:
  - a student answer with no table read is graded correct (verifier breach),
  - evaluation pass@1 is at least --collapse below both the first and the best
    evaluation for two evaluations in a row,
  - status.json has not changed for --hang-minutes while the process is alive,
  - sampled-token NLL stays above 5x its early median for 3 steps (policy noise).
It records, without stopping: skipped nonfinite updates, high MLX peak memory,
falling training reward, a held-out selection score below its best, and a trainer
exit without the done phase.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import read_json, read_jsonl

RUNS = Path(__file__).resolve().parents[1] / "runs"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def check(run, metrics, evals, status, args, now):
    alerts, stop = [], None
    if metrics:
        last = metrics[-1]
        if last.get("update_skipped_nonfinite"):
            alerts.append(f"step {last['step']}: nonfinite update skipped")
        if last["peak_mem_gb"] > args.mem_alert:
            alerts.append(f"step {last['step']}: MLX peak {last['peak_mem_gb']:.1f} GiB")
        if len(metrics) >= 20:
            early = sum(m["reward_mean"] for m in metrics[:10]) / 10
            recent = sum(m["reward_mean"] for m in metrics[-10:]) / 10
            if recent < early - 0.3:
                alerts.append(f"training reward fell from {early:.3f} to {recent:.3f}")
        nll = [m["sample_nll"] for m in metrics if "sample_nll" in m]
        if len(nll) >= 8:
            base = sorted(nll[:5])[2]
            if nll[-1] > 3 * base:
                alerts.append(f"sampled-token NLL {nll[-1]:.3f}, early median {base:.3f}")
            if all(v > 5 * base for v in nll[-3:]):
                stop = f"policy noise: sampled-token NLL above 5x its early median ({base:.3f}) for 3 steps"
        samples = read_json(run / "samples" / f"step_{last['step']:05d}.json", [])
        breach = [s for s in samples if s["source"] == "student" and s["outcome"] == "correct" and s["literal"]]
        if breach:
            stop = f"step {last['step']}: {len(breach)} literal answers graded correct"
    held = [e["by_split"]["heldout_select"] for e in evals if "heldout_select" in e.get("by_split", {})]
    if len(held) >= 2 and held[-1] < max(held[:-1]) - args.collapse:
        alerts.append(f"held-out selection {held[-1]:.3f}, best {max(held[:-1]):.3f}; the trainer stops early on a sustained drop")
    if len(evals) >= 3:
        best = max(e["pass1"] for e in evals[:-1])
        floor = min(best, evals[0]["pass1"]) - args.collapse
        if evals[-1]["pass1"] < floor and evals[-2]["pass1"] < floor:
            stop = f"evaluation collapse: {evals[-2]['pass1']:.3f}, {evals[-1]['pass1']:.3f} vs first {evals[0]['pass1']:.3f}, best {best:.3f}"
    if status and status["phase"] not in ("done", "error") and now - status["updated"] > args.hang_minutes * 60:
        stop = f"no status update for {(now - status['updated']) / 60:.0f} minutes in phase {status['phase']}"
    return alerts, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--collapse", type=float, default=0.15)
    ap.add_argument("--hang-minutes", type=float, default=45)
    ap.add_argument("--mem-alert", type=float, default=24)
    args = ap.parse_args()
    run = RUNS / args.run
    out = run / "monitor.jsonl"
    stopped = False
    while not run.exists() and alive(args.pid):
        time.sleep(5)  # the trainer creates the run directory after loading MLX
    if not run.exists():
        return
    while True:
        now = time.time()
        status, metrics, evals = read_json(run / "status.json"), read_jsonl(run / "metrics.jsonl"), read_jsonl(run / "evals.jsonl")
        is_alive = alive(args.pid)
        alerts, stop = check(run, metrics, evals, status, args, now)
        row = {"time": now, "alive": is_alive, "phase": status and status["phase"], "step": status and status["step"],
               "reward_mean": metrics[-1]["reward_mean"] if metrics else None,
               "outcomes": metrics[-1]["outcomes"] if metrics else None,
               "peak_mem_gb": metrics[-1]["peak_mem_gb"] if metrics else None,
               "step_s": metrics[-1]["rollout_s"] + metrics[-1]["train_s"] if metrics else None,
               "eval": {k: evals[-1].get(k) for k in ("step", "pass1", "by_split")} if evals else None,
               "alerts": alerts, "stop": stop}
        if stop and is_alive and not stopped:
            os.kill(args.pid, signal.SIGTERM)
            stopped = True
            row["action"] = "SIGTERM"
        if not is_alive:
            row["final"] = "done" if status and status["phase"] == "done" else "exited without the done phase"
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if not is_alive:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
