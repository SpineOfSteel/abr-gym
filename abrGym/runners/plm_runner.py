from __future__ import annotations
import subprocess
import sys


def run_plm_test(args, meta, paths) -> None:
    if not args.model_dir:
        raise ValueError("--model-dir is required for PLM test")
    if not args.exp_pool_path:
        raise ValueError("--exp-pool-path is required for PLM test")

    cmd = [
        sys.executable,
        str(paths.llm_entry),
        "--test",
        "--plm-type", meta["plm_type"],
        "--plm-size", args.plm_size,
        "--rank", str(args.rank),
        "--device", args.device,
        "--model-dir", args.model_dir,
        "--exp-pool-path", args.exp_pool_path,
    ]

    subprocess.run(cmd, check=True)