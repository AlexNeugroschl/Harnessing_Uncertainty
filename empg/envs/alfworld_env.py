"""Real ALFWorld (text/TextWorld version), one env per game file.

GRPO groups need N rollouts of the *same* game, so instead of AlfredTWEnv's
random batched sampler we register one TextWorld gym env per game file using
ALFWorld's own wrappers. Game discovery applies the same filters as
``AlfredTWEnv.collect_game_files`` (no movable/sliced, solvable only).

Data: ``alfworld-download`` puts it under $ALFWORLD_DATA (default ~/.cache/alfworld).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from empg.prompts import alfworld_prompt

SPLITS = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}
TASK_TYPES = {  # ALFWorld dir prefix -> paper's subtask column
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}
_BANNER = "-= Welcome to TextWorld, ALFRED! =-"


def _data_root(data_dir: str | None) -> Path:
    root = data_dir or os.environ.get("ALFWORLD_DATA") or "~/.cache/alfworld"
    return Path(os.path.expandvars(root)).expanduser()


def collect_games(split: str, data_dir: str | None = None) -> list[str]:
    """Solvable game files for a split, sorted (cached next to the data)."""
    root = _data_root(data_dir)
    cache = root / f"empg_games_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    games = []
    for game in sorted((root / "json_2.1.1" / SPLITS[split]).rglob("game.tw-pddl")):
        d = str(game.parent)
        if "movable" in d or "Sliced" in d or not (game.parent / "traj_data.json").exists():
            continue
        if not json.loads(game.read_text()).get("solvable", False):
            continue
        games.append(str(game))
    if not games:
        raise FileNotFoundError(f"no ALFWorld games under {root}; run `alfworld-download`")
    cache.write_text(json.dumps(games))
    return games


class ALFWorldEnv:
    def __init__(self, game_file: str, max_episode_steps: int = 200):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file], infos, batch_size=1, asynchronous=False,
            max_episode_steps=max_episode_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(env_id)
        self.task = ""

    @staticmethod
    def _info(infos) -> dict:
        adm = [a for a in infos["admissible_commands"][0] if a != "help"]
        return {"admissible": adm, "won": bool(infos["won"][0])}

    def reset(self):
        obs, infos = self.env.reset()
        obs = obs[0].replace(_BANNER, "").strip()
        self.task = obs.split("Your task is to: ")[-1].strip().rstrip(".")
        info = self._info(infos)
        info["task"] = self.task
        return obs, info

    def step(self, action: str):
        obs, scores, dones, infos = self.env.step([action])
        info = self._info(infos)
        info["task"] = self.task
        return obs[0], float(scores[0]), bool(dones[0]) or info["won"], info

    def close(self):
        self.env.close()


class ALFWorldBenchmark:
    name = "alfworld"

    def __init__(self, data_dir: str | None = None, history_length: int = 2):
        self.data_dir = data_dir
        self.history_length = history_length

    def tasks(self, split: str) -> list[str]:
        return collect_games(split, self.data_dir)

    def make_env(self, task: str) -> ALFWorldEnv:
        return ALFWorldEnv(task)

    def task_type(self, task: str) -> str:
        trial_dir = Path(task).parent.parent.name  # e.g. pick_heat_then_place_in_recep-Egg-None-...
        for prefix, name in TASK_TYPES.items():
            if trial_dir.startswith(prefix):
                return name
        return "Other"

    def prompt(self, task_desc, observation, admissible, history):
        return alfworld_prompt(task_desc, observation, admissible, history, self.history_length)
