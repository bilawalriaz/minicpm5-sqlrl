"""Measure MLX rollout and gradient batch sizes without changing model weights."""
import argparse
import json
import time
import subprocess
import re
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

import env
from rollout import rollout, set_memory_budget
from train import train_step


def swap_mib():
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    return float(match[1]) * (1024 if match[2] == "G" else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(Path(__file__).parent.parent / 'models/MiniCPM5-1B'))
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--prompts', type=int, default=16)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--gen-batches', type=int, nargs='+', default=[16, 32, 64])
    ap.add_argument('--micro-batches', type=int, nargs='+', default=[8192, 16384, 32768])
    ap.add_argument('--out', type=Path, default=Path('engineering/hardware-benchmark.json'))
    args = ap.parse_args()
    swap_start = swap_mib()
    set_memory_budget(26)
    model, tok = load(args.model)
    model.freeze()
    load_adapters(model, args.adapter)
    model.eval()
    tasks = [t for t in env.sample_tasks(args.prompts, seed=2026) for _ in range(args.group)]
    measurements = []
    episodes = None
    for gen_batch in args.gen_batches:
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.random.seed(42)
        eps, stats = rollout(model, tok, tasks, batch_size=gen_batch)
        row = {'phase': 'rollout', 'gen_batch': gen_batch, 'episodes': len(eps), **stats,
               'tokens_per_second': stats['gen_tokens'] / stats['seconds'],
               'peak_mem_gib': mx.get_peak_memory() / 1024**3,
               'correct': sum(e.outcome == 'correct' for e in eps)}
        measurements.append(row)
        print(json.dumps(row), flush=True)
        # Use the same trajectories for all gradient measurements.
        if episodes is None:
            episodes = eps
    batch = [(e.tokens, e.mask, 1.0 if i % 2 else -1.0) for i, e in enumerate(episodes)]
    for micro_tokens in args.micro_batches:
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.monotonic()
        # LR zero measures forward/backward/optimizer work without training.
        loss, grad, tokens = train_step(model, optim.Adam(learning_rate=0.0), batch, 1.0, micro_tokens, 1.0)
        seconds = time.monotonic() - started
        row = {'phase': 'gradient', 'micro_tokens': micro_tokens, 'seconds': seconds,
               'tokens': tokens, 'tokens_per_second': tokens / seconds, 'loss': loss,
               'grad_norm': grad, 'peak_mem_gib': mx.get_peak_memory() / 1024**3}
        measurements.append(row)
        print(json.dumps(row), flush=True)
    result = {'hardware': mx.device_info(), 'verifier_version': env.VERIFIER_VERSION,
              'protocol_version': env.PROTOCOL_VERSION, 'episodes': len(tasks),
              'adapter': str(Path(args.adapter).relative_to(Path.cwd())) if Path(args.adapter).is_absolute() else args.adapter,
              'swap_mib_before': swap_start, 'swap_mib_after': swap_mib(),
              'measurements': measurements}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
