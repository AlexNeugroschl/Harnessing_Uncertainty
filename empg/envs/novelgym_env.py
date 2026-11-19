"""Controlled crafting tasks on the real NovelGridWorldsV2 transition engine.

This is a custom text benchmark, not the published NovelGym evaluation panel.
Uses native ConfigParser, crafting, approach, selection, and breaking actions.
The single-agent loop avoids rendering, sockets, planners, and AEC bookkeeping.
"""

from functools import lru_cache
import hashlib
import json
import math
import re

import numpy as np

from empg.prompts import parse_action

UPSTREAM_REVISION = "f5d06d4f5166ca985ca420c809129a79700d904e"
SUITE_VERSION = 3  # legacy protocol remains available for existing artifacts
RECOVERY_SUITE_VERSION = 4
PREFIX = "gym_novel_gridworlds2.contrib.polycraft."


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


@lru_cache(maxsize=32)
def task_config(depth):
    """An independently specified recipe chain with a verifiable solution.

    One log produces one plank. Each component consumes a plank and the previous
    component. Increasing depth adds causal dependencies, not just a turn cap.
    Approach is a native macro action and counts as one turn in ALL conditions.
    """
    def recipe(inputs, output):
        return {"input": inputs, "output": {output: 1}, "step_cost": 0}

    recipes = {"planks": recipe(["log"], "planks"), "brace": recipe(["planks"], "brace")}
    for i in range(1, depth + 1):
        name = "pogo_stick" if i == depth else f"component_{i:02d}"
        inputs = ["planks"] + ([f"component_{i - 1:02d}"] if i > 1 else [])
        recipes[name] = recipe(inputs, name)
    size = max(10, math.ceil(math.sqrt(5 * (depth + 4))) + 2)
    objects = {key: {"quantity": depth + 4, "room": 1, "chunked": False}
               for key in ("log", "birch_log")}
    return {
        "map_size": [size, size], "rooms": {"1": {"start": [0, 0], "end": [size - 1, size - 1]}},
        "object_types": {
            "default": PREFIX + "objects.PolycraftObject",
            "bedrock": PREFIX + "objects.UnbreakablePolycraftObject",
            "log": PREFIX + "objects.BreakablePolycraftObject",
            "birch_log": PREFIX + "objects.BreakablePolycraftObject",
            "axe": PREFIX + "objects.PolycraftObject",
        },
        "objects": objects,
        "actions": {name: {"module": PREFIX + "actions." + cls, "step_cost": 0}
                    for name, cls in (("break_block", "Break"), ("nop", "NOP"),
                                      ("trade", "Trade"), ("craft", "Craft"))},
        "action_sets": {"main": ["approach_log", "approach_birch_log", "select_axe",
                                  "break_block", "craft_*", "nop"]},
        "entities": {"main_1": {
            "agent": "gym_novel_gridworlds2.agents.agent.Agent", "name": "Player", "type": "agent",
            "entity": PREFIX + "objects.PolycraftEntity", "action_set": "main",
            "inventory": {"axe": 1}, "id": 0, "room": 1, "max_step_cost": 1000000}},
        "recipes": recipes, "trades": {}, "auto_pickup_agents": [0],
    }


def nominal_solution_length(depth, novelty="none", regime="baseline"):
    # 2 actions per harvested log + one plank craft + one component craft.
    # Tool selection adds one; the changed final recipe adds one harvest/plank/brace.
    extra = 1 if regime == "novelty" and novelty == "tool" else 4 if regime == "novelty" and novelty == "recipe" else 0
    return 4 * depth + extra


class NovelGymEnv:
    def __init__(self, task, *, novelty="tool", regime="baseline", change="rules", feedback="generic"):
        self.task_id = task
        _, depth, seed = task.split(":")
        self.depth, self.seed = int(depth), int(seed)
        self.novelty, self.regime, self.change, self.feedback = novelty, regime, change, feedback
        if novelty not in ("tool", "recipe", "resource") or change not in ("rules", "rewards"):
            raise ValueError("invalid NovelGym novelty/change")
        if feedback not in ("generic", "explicit"):
            raise ValueError("invalid NovelGym feedback")
        if regime not in ("baseline", "novelty"):
            raise ValueError("NovelGym regimes must be baseline or novelty")
        if change == "rewards" and novelty != "tool":
            raise ValueError("reward-only intervention uses novelty=tool (selected-tool completion route)")
        self.state = None

    def reset(self):
        try:
            from gym_novel_gridworlds2.utils.json_parser import ConfigParser
        except ImportError as error:
            raise RuntimeError("Install requirements-novelgym.txt in the experiment Python environment") from error
        # ConfigParser owns a deep copy; its parsing mutates recipe/action dicts.
        self.state, self.dynamic, manager = ConfigParser().parse_json(
            json_content=task_config(self.depth), rng=np.random.default_rng(self.seed), rendering=False)
        rep = manager.get_agent("agent_0")
        self.entity, self.action_set = rep.entity, rep.action_set
        self.actions = {name: i for i, (name, _) in enumerate(self.action_set.actions)}
        self.done, self.unprepared = False, False
        if self.regime == "novelty" and self.change == "rules":
            if self.novelty == "tool":
                for name in ("log", "birch_log"):
                    for obj in self.state.get_objects_of_type(name):
                        obj.breakable, obj.breakable_holding = False, ["axe"]
            elif self.novelty == "recipe":
                recipe = self.dynamic.recipe_set.get_recipe("pogo_stick")
                recipe.input_list.append("brace")
                recipe.input_dict["brace"] = 1
            elif self.novelty == "resource":
                recipe = self.dynamic.recipe_set.get_recipe("planks")
                recipe.input_list[:] = ["birch_log"]
                recipe.input_dict = {"birch_log": 1}
        return self._observation("Ready."), self._info(False, 0)

    def _visible(self):
        # Observable resource counts and local view. Hidden object properties,
        # recipes, controller regime, RNG and time never enter curiosity keys.
        resources = {name: len(self.state.get_objects_of_type(name)) for name in ("log", "birch_log")}
        delta = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}[self.entity.facing]
        loc = tuple(int(a + b) for a, b in zip(self.entity.loc, delta))
        ahead = self.state.get_objects_at(loc)[0]
        return {"position": list(map(int, self.entity.loc)), "facing": self.entity.facing,
                "selected": self.entity.selectedItem,
                "route_prepared": not self.unprepared,
                "inventory": {k: int(v) for k, v in sorted(self.entity.inventory.items()) if v > 0},
                "resources": resources, "ahead": ahead[0].type if ahead else "air"}

    def _observation(self, feedback):
        visible = self._visible()
        # Only this visible dictionary is hashed, so unavailable world facts
        # cannot silently supply an oracle exploration signal.
        return feedback + "\n" + json.dumps(visible, separators=(",", ":"))

    def _info(self, won, failures):
        visible = self._visible()
        # Progress novelty deliberately ignores position/facing/resource layout.
        # Both fields are displayed in the observation; no hidden rule is used.
        progress = {k: visible[k] for k in ("inventory", "selected")}
        return {"task": f"Craft one pogo_stick through {self.depth} dependent component recipes.",
                "admissible": list(self.actions), "won": won,
                "state_keys": {"parsed": _digest(visible), "progress": _digest(progress)},
                "intervention_metrics": {"action_failures": failures}}

    def step(self, action):
        from gym_novel_gridworlds2.actions.action import PreconditionNotMetError
        if self.done:
            raise RuntimeError("step after terminal state")
        failed, feedback = False, "Operation succeeded."
        before = dict(self.entity.inventory)
        previous_selection = self.entity.selectedItem
        if action not in self.actions:
            failed, feedback = True, "Unknown action."
        else:
            try:
                op = self.action_set.actions[self.actions[action]][1]
                op.do_action(self.entity)
                if action == "break_block" and self.entity.selectedItem != "axe":
                    self.unprepared = True
            except PreconditionNotMetError as error:
                failed = True
                feedback = str(error) if self.feedback == "explicit" else "Operation failed."
        self.state.time_updates()
        completed = bool(self.state._goal_achieved)
        won = completed and not (self.regime == "novelty" and self.change == "rewards" and self.unprepared)
        self.done = completed
        if completed:
            feedback = "Task completed. Reward received." if won else "Task completed. No reward received."
        obs, info = self._observation(feedback), self._info(won, int(failed))
        info["intervention_metrics"]["base_won"] = int(completed)
        harvested = sum(max(0, self.entity.inventory.get(k, 0) - before.get(k, 0)) for k in ("log", "birch_log"))
        component = max([0, *(int(k.removeprefix("component_")) for k, v in self.entity.inventory.items()
                             if k.startswith("component_") and v > 0), self.depth if completed else 0])
        info["diagnostics"] = {"failed": failed,
            "tool_selected": int(not failed and self.entity.selectedItem == "axe" and previous_selection != "axe"),
            "harvested_items": harvested, "successful_harvest": int(action == "break_block" and harvested > 0),
            "crafted_items": int(action.startswith("craft_") and not failed), "component_depth": component}
        return obs, float(won), self.done, info

    def close(self):
        self.state = None
        self.dynamic = self.entity = self.action_set = None


class NovelGymBenchmark:
    name = "novelgym"

    def __init__(self, recipe_depth=8, novelty="tool", tasks_per_split=64, history_length=2,
                 workers=0, action_parser="strict", intervention=None, suite_version=SUITE_VERSION,
                 upstream_revision=UPSTREAM_REVISION, memory=False):
        if (not isinstance(recipe_depth, int) or not 2 <= recipe_depth <= 32
                or not isinstance(tasks_per_split, int) or tasks_per_split < 1
                or not isinstance(history_length, int) or history_length < 0 or workers < 0):
            raise ValueError("invalid NovelGym depth/task/history/worker count")
        if novelty not in ("tool", "recipe", "resource"):
            raise ValueError("novelty must be tool, recipe or resource")
        if memory or workers:
            raise ValueError("NovelGym uses compact current-state prompts and serial native stepping; set memory=false, workers=0")
        if suite_version not in (SUITE_VERSION, RECOVERY_SUITE_VERSION) or upstream_revision != UPSTREAM_REVISION:
            raise ValueError("unsupported NovelGym suite/upstream version")
        self.depth, self.novelty, self.count = recipe_depth, novelty, tasks_per_split
        self.suite_version = suite_version
        self.history_length, self.workers, self.action_parser = history_length, workers, action_parser
        self.intervention = intervention
        if action_parser not in ("strict", "verl", "native"):
            raise ValueError("NovelGym action_parser must be native, strict or verl")

    def tasks(self, split):
        # Reserve a fresh confirmation range after the 2M engineering probes.
        offsets = {"train": 0, "valid_seen": 1000000, "valid_unseen": 3000000}
        if self.suite_version == RECOVERY_SUITE_VERSION:
            offsets = {"train": 100000, "valid_seen": 1500000, "valid_unseen": 4000000}
        if split not in offsets:
            raise ValueError("unknown task split")
        return [f"craft:{self.depth}:{offsets[split] + i}" for i in range(self.count)]

    def task_type(self, task):
        return f"depth_{int(task.split(':')[1])}"

    def make_env(self, task):
        spec = self.intervention or {"regime": "baseline", "change": "rules", "feedback": "generic"}
        return NovelGymEnv(task, novelty=self.novelty, **{k: spec[k] for k in ("regime", "change", "feedback") if k in spec})

    def episode_batch(self, tasks):
        from empg.envs.batch import SerialBatch
        # Native lightweight Python actions cost far less than generation.
        # Avoid IPC and worker CUDA initialization on this single GPU workload.
        return SerialBatch(self, tasks)

    def close(self):
        pass

    def parse_action(self, response):
        if self.action_parser == "native":
            tagged = re.findall(r"<action>\s*([a-z][a-z0-9_]*)\s*</action>", response, re.I)
            lines = re.findall(r"^\s*action\s*:\s*[`'\"]?([a-z][a-z0-9_]*)[`'\"]?\s*$", response, re.I | re.M)
            choices = tagged + lines
            if len(choices) == 1:
                return choices[0].lower(), True
            if not choices and re.fullmatch(r"[a-z][a-z0-9_]*", response.strip(), re.I):
                return response.strip().lower(), True
            return "nop", False
        return parse_action(response, mode=self.action_parser)

    def prompt(self, task_desc, observation, admissible, history):
        recent = history[-self.history_length:] if self.history_length else []
        # Baseline guide is identical across regimes; changed rules must be
        # inferred through actual action feedback, not leaked via configuration.
        recipes = task_config(self.depth)["recipes"]
        guide = "; ".join(f"{name}<-{'+'.join(r['input'])}" for name, r in recipes.items())
        past = "\n".join(f"{obs.splitlines()[0]} Action: {action}" for obs, action in recent)
        skill_guide = ("Initial skill guide: when you need a log and ahead is air, approach_log; "
                       "when ahead is log, break_block; when inventory contains a log, craft_planks. "
                       "Choose from the current inventory and feedback; these describe only the initial mechanics.\n")
        if self.suite_version == RECOVERY_SUITE_VERSION:
            skill_guide = ("The guide describes initial mechanics. Feedback takes priority: after an operation fails, "
                           "reconsider its prerequisites and try a different applicable action. "
                           "Follow the inventory rather than continuing a failed crafting sequence.\n")
        return (f"{task_desc}\nKnown initial recipes: {guide}. Each recipe yields one item. "
                "approach_log faces a log; approach_birch_log faces a birch log. Approaching does not collect anything. "
                "break_block harvests the resource directly ahead into your inventory. "
                "craft_planks consumes a harvested log and produces one plank. "
                "select_axe equips your existing axe. Build the ingredients of missing components in order. "
                "Actions can fail; use feedback to revise your plan.\n"
                + skill_guide +
                f"{past}\n{observation}\nActions: {', '.join(admissible)}\n"
                + ("Return exactly one line: action: ACTION_NAME. Replace ACTION_NAME with one exact name from Actions."
                   if self.action_parser == "native" else
                   "Respond with exactly one short thought and one action: <think>reason</think><action>exact_action_name</action>."))
