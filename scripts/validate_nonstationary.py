"""Real-game solvability, worker parity, and CPU overhead. No model load.

Run from the repository root in the ALFWorld environment:
    python -m scripts.validate_nonstationary --games 6 --out results/nonstationary_validation.json
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

from empg.envs.alfworld_env import ALFWorldBenchmark, ALFWorldEnv
from empg.nonstationary import InspectionEnv, NonstationaryConfig


class ExpertEnv(ALFWorldEnv):
    def _info(self, infos):
        self.expert_state = {k: v[0] for k, v in infos.items()}
        return super()._info(infos)

    def reset(self):
        obs, info = super().reset()
        self.expert_state["feedback"] = "-= Welcome to TextWorld, ALFRED! =-\n" + obs
        return obs, info

    def step(self, action):
        result = super().step(action)
        self.expert_state["feedback"] = result[0]
        return result


def expert_solution(task, seed):
    from alfworld.agents.expert.handcoded_expert_tw import HandCodedTWAgent
    random.seed(seed)
    env, agent, solution = ExpertEnv(task), HandCodedTWAgent(max_steps=180), []
    try:
        obs, info = env.reset()
        agent.reset(task)
        last_action = ""
        for _ in range(180):
            action = agent.act(env.expert_state, 0., False, last_action)
            obs, reward, done, info = env.step(action)
            solution.append(action)
            last_action = action
            if done:
                return solution if info["won"] else None
    finally:
        env.close()
    return None


def replay(task, solution, change, regime, prepared, feedback="explicit"):
    env = InspectionEnv(ALFWorldEnv(task), **NonstationaryConfig(change=change, feedback=feedback).intervention(regime))
    steps, blocks, reward = 0, 0, 0.
    try:
        env.reset()
        for action in solution:
            target = env._target(action)
            if prepared and target and action in env.native_actions and target not in env.inspected:
                env.step(f"inspect {target}")
                steps += 1
            _, r, done, info = env.step(action)
            reward += r
            steps += 1
            blocks += info["intervention_metrics"].get("rule_blocks", 0)
            if done:
                break
        return {"won": bool(info.get("won")), "env_return": reward, "steps": steps, "rule_blocks": blocks,
                "base_won": bool(info["intervention_metrics"].get("base_won", False))}
    finally:
        env.close()


def worker_parity(task, solution, feedback="explicit"):
    def equal(left, right):
        # Worker JSON transport normalizes tuples to lists.
        assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), (left, right)
    spec = NonstationaryConfig(feedback=feedback).intervention("inspection")
    serial, parallel = ALFWorldBenchmark(workers=0, intervention=spec), ALFWorldBenchmark(workers=2, intervention=spec)
    a, b = serial.episode_batch([task, task]), parallel.episode_batch([task, task])
    try:
        equal(a.reset(), b.reset())
        for action in solution:
            # Insert an inspection whenever a selected command is locally valid.
            targets = a.envs[0].targets
            target = a.envs[0]._target(action)
            if target in targets and target not in a.envs[0].inspected:
                inspect = f"inspect {target}"
                equal(a.step([0, 1], [inspect, inspect]), b.step([0, 1], [inspect, inspect]))
            left, right = a.step([0, 1], [action, action]), b.step([0, 1], [action, action])
            equal(left, right)
            if left[0][2]:
                break
        # Exercise a phase switch on the SAME worker pool.
        b.close()
        parallel.intervention = NonstationaryConfig(feedback=feedback).intervention("baseline")
        b = parallel.episode_batch([task, task])
        expected = ALFWorldBenchmark(workers=0, intervention=parallel.intervention).episode_batch([task, task])
        try:
            equal(expected.reset(), b.reset())
        finally:
            expected.close()
    finally:
        a.close()
        b.close()
        parallel.close()
        serial.close()


def overhead(repeats=20000):
    class Native:
        def reset(self):
            return "room", {"admissible": ["take apple 1 from table 1", "look"],
                            "state_keys": {"parsed": "room"}, "won": False}

        def step(self, action):
            obs, info = self.reset()
            return obs, 0., False, info

    timings = {}
    for name, env in (("native", Native()), ("intervention", InspectionEnv(
            Native(), **NonstationaryConfig().intervention("baseline")))):
        env.reset()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                env.step("look")
            samples.append((time.perf_counter() - start) / repeats)
        timings[f"{name}_microseconds_per_step"] = statistics.median(samples) * 1e6
    timings["added_microseconds_per_step"] = timings["intervention_microseconds_per_step"] - timings["native_microseconds_per_step"]
    return timings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--out", default="results/nonstationary_validation.json")
    ap.add_argument("--feedback", choices=("explicit", "generic"), default="explicit")
    args = ap.parse_args()
    bench = ALFWorldBenchmark()
    # Prefer one game from each subtask, with fallback candidates when the
    # heuristic expert cannot solve a particular game.
    buckets = {}
    for task in bench.tasks("valid_seen"):
        buckets.setdefault(bench.task_type(task), []).append(task)
    results, first = [], None
    for typ, tasks in sorted(buckets.items()):
        if len(results) >= args.games:
            break
        solution = None
        for task in tasks[:8]:
            try:
                solution = expert_solution(task, 0)
            except Exception as error:
                print(f"Expert candidate failed: {typ}: {error}", flush=True)
            if solution:
                break
        if not solution:
            raise RuntimeError(f"No expert solution found for {typ}")
        first = first or (task, solution)
        cases = {}
        for change in ("rules", "rewards"):
            for regime in ("baseline", "inspection"):
                for prepared in (False, True):
                    key = f"{change}/{regime}/{'prepared' if prepared else 'original'}"
                    cases[key] = replay(task, solution, change, regime, prepared, args.feedback)
                    expected_win = regime == "baseline" or prepared
                    assert cases[key]["won"] == expected_win, (typ, key, cases[key])
                    if change == "rewards":
                        assert cases[key]["base_won"], (typ, key)
        results.append({"type": typ, "task": task, "native_solution_steps": len(solution), "cases": cases})
        print(f"Validated {typ}: {len(solution)} native actions, 8 intervention cases", flush=True)
    if not results:
        raise RuntimeError("No real games validated")
    worker_parity(*first, feedback=args.feedback)
    output = {"games": results, "feedback": args.feedback, "worker_parity": True, "cpu_overhead": overhead()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["cpu_overhead"], indent=2))


if __name__ == "__main__":
    main()
