"""Evaluate a local base/LoRA policy with the official AdaExploreBench runtime.

Requires an explicit pinned upstream checkout and its released dataset. Several
episodes share each GPU generation batch; one engine session spans evaluation.
This measures in-context rule adaptation, separately from crafting RL recovery.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from run_benchmarks import ROOT, launch_in_wsl

REVISION = "7ac28a364ed19b0461f5a987f8dd81fb57bd0567"


@contextmanager
def official_runtime(source):
    """Use upstream prompts and imports without copying benchmark semantics."""
    source = Path(source).resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"AdaExploreBench checkout must be at {REVISION}; found {actual}")
    clean = subprocess.run(["git", "-c", "core.autocrlf=true", "-c", "core.filemode=false",
                            "-C", str(source), "diff", "--quiet", "HEAD", "--"])
    if clean.returncode:
        raise ValueError("AdaExploreBench tracked source files must be unchanged")
    previous, imports = Path.cwd(), list(sys.path)
    sys.path[:0] = [str(source), str(source / "src")]
    os.chdir(source)  # upstream reads its official prompt templates relative to cwd
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path[:] = imports


class LocalBatchClient:
    """Asynchronous episode requests coalesced into a single local GPU call."""
    def __init__(self, policy, response_type, greedy=True):
        self.policy, self.response_type, self.greedy = policy, response_type, greedy
        self.pending, self.scheduled = [], False
        self.config = SimpleNamespace(model=policy.cfg.model_name, conversation_protocol="deepseek_chat",
                                      provider_retry_limit=0, syntax_repair_limit=1)

    async def generate(self, *, request, schema, trace_context=None):
        if request.protocol != "deepseek_chat" or not request.messages:
            raise ValueError("local evaluation requires the official deepseek_chat conversation projection")
        future = asyncio.get_running_loop().create_future()
        self.pending.append((request.messages, future))
        if not self.scheduled:
            self.scheduled = True
            asyncio.get_running_loop().call_soon(self.flush)
        return await future

    def flush(self):
        pending, self.pending = self.pending, []
        self.scheduled = False
        live = [(messages, future) for messages, future in pending if not future.cancelled()]
        if not live:
            return
        try:
            prompts, responses, texts = self.policy.generate([m for m, _ in live], greedy=self.greedy)
            if any(len(rows) != len(live) for rows in (prompts, responses, texts)):
                raise ValueError("local generation returned an incomplete batch")
            for (_, future), prompt, response, text in zip(live, prompts, responses, texts):
                future.set_result(self.response_type(content=text, model=self.config.model,
                    usage={"prompt_tokens": len(prompt), "completion_tokens": len(response),
                           "total_tokens": len(prompt) + len(response)}, raw={"local_generation": True},
                    finish_reason="length" if self.policy.truncated(response) else "stop"))
        except Exception as error:
            for _, future in live:
                if not future.done():
                    future.set_exception(error)


async def evaluate_episodes(episodes, client, output, *, batch_envs, conditions, branch, baseline=False, seed=0):
    from runtime.runner import run_agent_episode, run_baseline_episode
    from runtime.metrics import aggregate_metrics
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    # Conditions are paired on exactly the same released episodes/timeline.
    for condition in conditions:
        for start in range(0, len(episodes), batch_envs):
            chunk = episodes[start:start + batch_envs]
            async def run(episode):
                if baseline:
                    artifact = run_baseline_episode(episode, policy="random", condition=condition,
                        seed=seed,
                        sampling_branch=branch, history_policy="eval_case_hidden",
                        interaction_policy={"max_probe_cases_per_batch": 16, "allow_abstention": True})
                else:
                    artifact = await run_agent_episode(episode, client, condition=condition,
                        sampling_branch=branch, history_policy="eval_case_hidden",
                        interaction_policy={"max_probe_cases_per_batch": 16, "allow_abstention": True})
                # Use ordinal names; release identifiers never become filesystem paths.
                index = start + chunk.index(episode)
                path = output / f"{condition}_{index:04d}.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
                tmp.replace(path)
                return artifact
            artifacts.extend(await asyncio.gather(*(run(ep) for ep in chunk)))
            print(f"{condition}: {min(start + batch_envs, len(episodes))}/{len(episodes)}", flush=True)
    indexed = {ep["episode_id"]: ep for ep in episodes}
    summary = {condition: aggregate_metrics([a for a in artifacts if a["condition"] == condition], indexed)
               for condition in conditions}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_screen", *argv], entrypoint="run_adaexplore.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adapter", help="trained LoRA directory; omit for the base model")
    ap.add_argument("--out", default="results/adaexplore_local")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--batch-envs", type=int, default=4)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--response-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--branch", default="flip_50")
    ap.add_argument("--conditions", nargs="+", choices=["dynamic", "static"], default=["dynamic", "static"])
    ap.add_argument("--runtime-check", action="store_true", help="official model-free random baseline; no GPU")
    ap.add_argument("--tier", default="novelgym_screen", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if min(args.max_episodes, args.batch_envs, args.response_tokens) < 1 or args.context <= args.response_tokens or args.seed < 0:
        ap.error("positive episode/batch/response limits and context > response limit required")
    if len(args.conditions) != len(set(args.conditions)):
        ap.error("conditions must be unique")
    dataset, source, out = (Path(v).resolve() for v in (args.dataset, args.source, args.out))
    if out.exists() and any(out.iterdir()):
        ap.error("choose a new or empty output directory to preserve prior evaluations")
    payload = dataset.read_bytes()
    episodes = [json.loads(line) for line in payload.decode().splitlines() if line.strip()][:args.max_episodes]
    if not episodes or len({ep["episode_id"] for ep in episodes}) != len(episodes):
        ap.error("dataset must contain nonempty unique episode IDs")
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    with official_runtime(source):
        if args.runtime_check:
            summary = asyncio.run(evaluate_episodes(episodes, None, out, batch_envs=args.batch_envs,
                        conditions=args.conditions, branch=args.branch, baseline=True, seed=args.seed))
            model_config = None
        else:
            from benchmarks.common import load
            from empg.policy import LLMPolicy
            from generator import GeneratorResponse
            cfg = load("grpo", args.tier)
            pc = dataclasses.replace(cfg.policy, max_prompt_tokens=args.context - args.response_tokens,
                max_new_tokens=args.response_tokens, vllm_max_model_len=args.context,
                vllm_max_num_seqs=args.batch_envs, gen_batch_size=args.batch_envs)
            policy = LLMPolicy(pc, seed=args.seed)
            try:
                if adapter:
                    policy.load_adapter(adapter)
                client = LocalBatchClient(policy, GeneratorResponse)
                with policy.sampling(args.seed), policy.generation():
                    summary = asyncio.run(evaluate_episodes(episodes, client, out, batch_envs=args.batch_envs,
                                conditions=args.conditions, branch=args.branch))
            finally:
                policy.close()
            model_config = dataclasses.asdict(pc)
    manifest = {"upstream_revision": REVISION, "dataset_sha256": hashlib.sha256(payload).hexdigest(),
                "adapter": adapter, "policy": model_config, "options": vars(args),
                "note": ("Official model-free random baseline; runtime integration check only." if args.runtime_check else
                         "Official environment/history/metrics with local greedy text generation and JSON parsing; no weight updates or intrinsic reward during evaluation.")}
    if adapter:
        manifest["adapter_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(adapter).iterdir()) if p.is_file() and p.suffix in (".json", ".safetensors", ".bin")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({condition: result.get("overall") for condition, result in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
