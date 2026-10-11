"""Real ALFWorld (text/TextWorld version), one env per game file.

GRPO groups need N rollouts of the *same* game, so instead of AlfredTWEnv's
random batched sampler we register one TextWorld gym env per game file using
ALFWorld's own wrappers. Game discovery applies the same filters as
``AlfredTWEnv.collect_game_files`` (no movable/sliced, solvable only).

Data: ``alfworld-download`` puts it under $ALFWORLD_DATA (default ~/.cache/alfworld).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from empg.prompts import alfworld_prompt, visited_locations

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
# TextWorld facts that record history, not the physical world: `checked(loc)`
# marks inspected locations, and explicit negations (`not_atlocation`,
# `not_holds`, `not_holdsany`, ...) appear once the agent has left a place or put
# something down. Keeping them would make every `look`, revisit or put-back a
# "new" state.
_HISTORY_FACTS = ("checked(", "not_")


def _digest(items) -> str:
    return hashlib.md5("\n".join(sorted(items)).encode()).hexdigest()[:16]


class ParsedState:
    """Non-privileged symbolic state, tracked from actions and their feedback.

    Location, inventory, where moved objects now sit (relative to where they were
    first taken from), opened receptacles, object conditions (clean/hot/cool) and
    switched-on appliances. An action counts only if the env did not answer
    "Nothing happens."

    Against the oracle physical key on random walks, a change in one key always
    coincides with a change in the other. The remaining difference is
    deliberate: ALFWorld puts several receptacles at one physical location
    (e.g. drawer 7 and drawer 8), which the oracle cannot tell apart but the
    agent sees as different places.
    """

    _RULES = [
        (re.compile(r"^go to (.+)$"), "goto"),
        (re.compile(r"^take (.+) from (.+)$"), "take"),
        (re.compile(r"^(?:move|put) (.+?) (?:to|in|on|in/on) (.+)$"), "put"),
        (re.compile(r"^open (.+)$"), "open"),
        (re.compile(r"^close (.+)$"), "close"),
        (re.compile(r"^(clean|heat|cool) (.+?) with (.+)$"), "condition"),
        (re.compile(r"^use (.+)$"), "use"),
    ]

    def __init__(self):
        self.location = "start"
        self.inventory: set[str] = set()
        self.origin: dict[str, str] = {}   # object -> receptacle it was first taken from
        self.placed: dict[str, str] = {}   # object -> receptacle it was last moved to
        self.opened: set[str] = set()
        self.conditions: set[str] = set()
        self.switched_on: set[str] = set()

    def update(self, action: str, observation: str):
        if observation.strip().startswith("Nothing happens"):
            return
        a = action.strip().lower()
        for pattern, kind in self._RULES:
            m = pattern.match(a)
            if not m:
                continue
            if kind == "goto":
                self.location = m.group(1)
            elif kind == "take":
                obj = m.group(1)
                self.origin.setdefault(obj, m.group(2))
                self.placed.pop(obj, None)
                self.inventory.add(obj)
            elif kind == "put":
                self.inventory.discard(m.group(1))
                self.placed[m.group(1)] = m.group(2)
            elif kind == "open":
                self.opened.add(m.group(1))
            elif kind == "close":
                self.opened.discard(m.group(1))
            elif kind == "condition":
                self.conditions.add(f"{m.group(1)}:{m.group(2)}")
            elif kind == "use":
                self.switched_on.add(m.group(1))
            return

    def key(self) -> str:
        return _digest([f"at:{self.location}",
                        *(f"inv:{x}" for x in self.inventory),
                        # Put back where it started = unchanged world, so omit it.
                        *(f"in:{o}@{r}" for o, r in self.placed.items() if r != self.origin.get(o)),
                        *(f"open:{x}" for x in self.opened),
                        *(f"cond:{x}" for x in self.conditions),
                        *(f"on:{x}" for x in self.switched_on)])


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

        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                                   extras=["gamefile"])
        # Instantiate directly: register_games would grow the global gym registry
        # by one entry for every episode in a long training run.
        from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv
        self.env = TextworldBatchGymEnv(
            [game_file], infos, batch_size=1, asynchronous=False,
            max_episode_steps=max_episode_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.task = ""
        self.parsed = ParsedState()

    def _info(self, infos) -> dict:
        adm = [a for a in infos["admissible_commands"][0] if a != "help"]
        facts = [str(f) for f in infos["facts"][0]]
        physical = [f for f in facts if not f.startswith(_HISTORY_FACTS)]
        checked = [f for f in facts if f.startswith("checked(")]
        return {
            "admissible": adm, "won": bool(infos["won"][0]), "task": self.task,
            "state_keys": {
                "parsed": self.parsed.key(),              # non-privileged
                "physical": _digest(physical),            # oracle: world state
                "knowledge": _digest(physical + checked),  # oracle: world + what was inspected
            },
        }

    def reset(self):
        obs, infos = self.env.reset()
        obs = obs[0].replace(_BANNER, "").strip()
        self.task = obs.split("Your task is to: ")[-1].strip().rstrip(".")
        self.parsed = ParsedState()
        return obs, self._info(infos)

    def step(self, action: str):
        obs, scores, dones, infos = self.env.step([action])
        self.parsed.update(action, obs[0])
        info = self._info(infos)
        return obs[0], float(scores[0]), bool(dones[0]) or info["won"], info

    def close(self):
        self.env.close()


class ALFWorldBenchmark:
    name = "alfworld"

    def __init__(self, data_dir: str | None = None, history_length: int = 2,
                 memory: bool = False, workers: int = 0):
        self.data_dir = data_dir
        self.history_length = history_length
        self.memory = memory  # list visited locations in the prompt (H6 baseline)
        if workers < 0:
            raise ValueError("env.workers must be nonnegative")
        self.workers = workers
        self._pool = None

    def episode_batch(self, tasks):
        from empg.envs.batch import ALFWorldPool, ParallelBatch, SerialBatch
        if not self.workers:
            return SerialBatch(self, tasks)
        if self._pool is None:
            self._pool = ALFWorldPool(min(self.workers, len(tasks)))
        return ParallelBatch(self._pool, tasks)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

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
        visited = visited_locations(history, observation) if self.memory else None
        return alfworld_prompt(task_desc, observation, admissible, history, self.history_length,
                               visited=visited)
