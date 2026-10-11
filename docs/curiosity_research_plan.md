# Research plan: curiosity for long-horizon, sparse-reward LLM agents

The October 11, 2026 [research audit](research_audit.md) narrows the theoretical claims, repairs the task-sampling
control, and adds paired evaluation plus a cheaper staged protocol. Use `pilot` on `valid_seen` for development;
reserve `valid_unseen` for a prespecified confirmation. Earlier timing estimates below predate the optimizations.

**Question.** Can an intrinsic "curiosity" signal help an LLM agent solve long-horizon tasks with sparse reward? The
signal rewards the agent for reaching states it hasn't tried before, and has to be balanced so that exploration
doesn't crowd out exploitation.

**Starting point.** This repo already has GRPO and EMPG ([arXiv 2509.09265](https://arxiv.org/pdf/2509.09265))
training Qwen2.5-1.5B + LoRA on real ALFWorld. Curiosity is added as another term in the same advantage pipeline,
so every experiment compares against GRPO and EMPG under identical conditions.

---

## Implementation status

| Plan item | Status | Where |
|---|---|---|
| State keys: parsed (non-privileged), physical and knowledge (oracle), obs | done | `empg/envs/alfworld_env.py` |
| Episodic / group count novelty, `1/√n` or first-visit bonus, cap | done | `empg/curiosity.py` |
| Step-level term and trajectory-level shaping (§3.3 options 1–2) | done | `compute_advantages(..., intrinsic=)` |
| Schedules (constant, linear, exp), flat-group gate, confidence modulation (§3.4) | done | `CuriosityConfig` |
| Memory-in-prompt baseline (H6) | done | `env.memory: true`, arm `grpo_memory` |
| Exploration metrics (§6), including a no-op rate that separates failed actions from real loops | done | `exploration_stats`, logged every iteration, `results/exploration.md` |
| Phase 0 script | done | `benchmarks/explore_stats.py` |
| Phase 2 arms A–H + size tiers | done | `benchmarks/configs/arms/`, `configs/tiers/`, `suite --phase screen` |
| Separate intrinsic/extrinsic normalisation (§3.3 option 3) | not yet | add only if Phase 2 shows the success signal being washed out |
| Rollout-level exploration, Go-Explore branching (§3.3 options 4–5) | not yet | stretch goals |
| Phase 5: Coin Collector, LLM surprisal, embedding / RND novelty, lifelong scope | not yet | optional |

Checks so far:
- All unit tests pass, including "η = 0 is bit-identical to GRPO/EMPG".
- Every new arm runs end to end at the `smoke` tier on GPU with real ALFWorld.

**Parsed vs oracle key.** On random walks over 12 games (415 physical states), every step that changes one
key also changes the other. The parsed key merges two different physical states only once, in an edge case
where two stove burners share one location.

Getting there needed two fixes:
- The parsed key now tracks where moved objects end up.
- The oracle now drops *all* `not_*` history facts, not only `not_atlocation`.

The parsed key does split some states the oracle merges (28 of 415), because ALFWorld puts several receptacles
(e.g. drawers 7 and 8) at one location. The agent sees those as different places, so this is intended.

Design note: the curiosity knobs live under `advantage.curiosity`, so arms still differ only in their
`advantage:` block. `grpo_memory` is the one exception, since it changes the prompt.

---

## 1. Why this is interesting next to EMPG

EMPG already has an intrinsic term: the future-clarity bonus `f(H_t+1) = exp(−k'·H_norm)`. It rewards actions
that lead to states **where the policy is confident**. Curiosity rewards actions that lead to states **the agent
hasn't seen**. These two measure different things:

| | Familiar state | Novel state |
|---|---|---|
| **Policy confident** | exploit (both signals agree) | curiosity says go; clarity says fine |
| **Policy uncertain** | clarity says avoid; curiosity says nothing new | **conflict**: curiosity says go, clarity says avoid |

So "curiosity vs clarity" is a real design axis, not a detail. It raises three questions:
- Do the two bonuses complement each other (novelty in *environment state*, confidence in *policy*)?
- Do they cancel out?
- Should the balance shift over training (explore early, then seek clarity)?

That comparison is a contribution in itself, and it's cheap to run because both live in `compute_advantages`.

**A known failure mode curiosity targets.** GRPO has no outcome contrast when every sibling fails. Its advantage
is zero only when the complete rewards also tie: this implementation penalizes invalid formats, so an all-failed
group can still learn formatting. `no_signal_frac` tracks binary outcome ties and `flat_group_frac` tracks actual
reward ties. Curiosity can do two things here:
- give those groups a learning signal
- more importantly, make it more likely that *at least one* rollout in the group succeeds, which turns a flat
  group into an informative one

---

## 2. Research questions and hypotheses

| # | Hypothesis | Main measurement |
|---|---|---|
| H1 | The base and early policy revisit states often (loops, re-checking the same drawer). Episodic novelty cuts revisits and raises success, mostly on the **long** subtasks (Pick2, Clean/Heat/Cool). | revisit rate; success per subtask |
| H2 | Curiosity mainly helps through **flat groups**: it raises P(group has ≥1 success) early in training. | `flat_group_frac` over time; P(≥1 success) |
| H3 | *Within-group* novelty (rewarding a rollout for going where its 7 siblings didn't) beats per-episode novelty for GRPO, because GRPO learns from differences between siblings. | same as H2, plus group state coverage |
| H4 | Curiosity and EMPG's clarity bonus are complementary: EMPG + curiosity > either one alone. | 2×2 (with/without each) success |
| H5 | A **schedule or gate** on curiosity (anneal it, or switch it on only in flat groups) beats a constant weight. Constant curiosity keeps exploring after the task is solvable. | final success; episode length of successes |
| H6 | Part of any curiosity gain is really **memory compensation**. The prompt only shows the last 2 steps, so the agent forgets where it has been. A prompt that lists visited locations may match curiosity-by-RL. | memory-in-prompt baseline vs curiosity |

H6 matters for honesty. If a "visited: desk 1, drawer 2" line in the prompt does as well as curiosity, the result
is about memory, not exploration.

---

## 3. Design space

### 3.1 What counts as "the same state"?

Novelty is only as good as the state key. ALFWorld offers a ladder from privileged to realistic:

| Key | Built from | Notes |
|---|---|---|
| **Physical state (oracle)** | TextWorld `facts=True`, dropping `checked(...)` and all `not_*` predicates | Privileged diagnostic for location, inventory and object conditions. It can merge distinct receptacles at one physical location, so it is not a guaranteed upper bound on parsed-key performance. |
| **Knowledge state (oracle)** | physical state + the set of `checked(loc)` facts | Counts *discovering* a location as progress. This fits ALFWorld well, where most of the work is finding the object. |
| **Parsed symbolic state** | location, inventory, placements, conditions and opened receptacles tracked from actions/feedback | No privileged access, but a lossy, hand-engineered state abstraction; random-walk checks are limited validation. **The default.** |
| **Observation hash** | normalised observation text | Simplest option. It breaks when the same state is described differently (a noisy TV in miniature) and when different states produce the same text ("Nothing happens."). |
| **Embedding novelty** | a sentence embedding of the observation (e.g. `all-MiniLM-L6-v2`, already in the HF cache), compared with kNN distance to episodic memory, or with RND prediction error | Generalises *across games*. Needed for lifelong novelty (see 3.2). |
| **LLM surprisal** | `−mean log p_ref(o_t+1 ∣ context, a_t)` scored by the frozen reference model (adapter off) | "LLM-native" curiosity with no state engineering: the agent's own language model acts as its world model. It costs one extra scoring pass. Its risk is that it rewards unusual *wording*, not new states. |

Plan: start with **parsed symbolic** and **physical oracle** (privileged diagnostic). Add embedding novelty and
LLM surprisal as the more novel, research-y options once the pipeline works.

### 3.2 Novelty over which scope?

| Scope | Count `n(s)` over | Formula idea | Comment |
|---|---|---|---|
| **Episodic** | this episode | `c_t = 1[first visit]` or `1/√n_ep(s_t+1)` | Directly targets loops (H1). Cheap. |
| **Group** | the sibling rollouts of the same game | `c_t = 1/√n_grp(s_t+1)`, where `n_grp` counts the siblings that reached `s` | Rewards within-group state coverage (H3). Novelty and GRPO already appear together in prior work; distinguish this state-based variant and its interaction with clarity rather than claiming the broad idea is new. |
| **Lifelong** | all of training | pseudo-counts / RND | **Caution:** with 3,553 training games and 2 per iteration, each game is seen about once. Per-game lifelong counts are therefore useless here. Lifelong novelty has to generalise *across* games, so it needs embeddings or an abstract key such as (receptacle type, held object type). |
| **Combined** (NGU-style) | episodic × lifelong multiplier | `c_t = c_ep · clip(m_life, 1, L)` | Standard in deep RL. Try it only if both parts help on their own. |

### 3.3 Where to inject it into the learning signal

All options plug into [empg/advantages.py](../empg/advantages.py) (`compute_advantages`), which is already the
only place advantages are made.

1. **Step-level advantage term (recommended default).** Mirrors EMPG's `ζ·f`:

   ```
   A_mod(i,t) = A_i · g(H_t) + ζ · f(H_t+1) + η · ĉ_t        ĉ = batch-normalised novelty (like Eq. 12)
   A_final    = A_mod − mean_B(A_mod)                          (Eq. 13 unchanged)
   ```

   It gives dense, per-step credit, and it is one extra array in the existing function.
2. **Trajectory-level reward shaping.** `r_i' = r_i + β · mean_t c_t`, then group normalisation. After Eq. 4,
   only a rollout's novelty *relative to its siblings* matters. That is an elegant fit to GRPO, but credit
   inside the trajectory stays uniform.
3. **Separate normalisation of the two streams.** Normalise the extrinsic and intrinsic advantages on their own,
   then mix them: `A = A_ext + η · A_int`. Deep-RL curiosity work (e.g. RND) keeps the two streams apart like
   this, so that the dense intrinsic signal can't swamp a rare success. This is worth testing if option 1 shows
   success being washed out.
4. **Rollout-level exploration (no reward change).** Examples: sampling temperature > 1 for early turns, or forcing
   the group's first actions to differ. **Caution:** sampling from a modified distribution breaks two
   assumptions. It breaks the PPO ratio's on-policy assumption, and it breaks EMPG's "entropy of the sampling
   policy". Treat these as baselines, not the main method.
5. **Go-Explore-style branching (stretch goal).** ALFWorld is deterministic, so any state can be restored by
   replaying its action prefix. The idea is to branch the group from a *frontier* state (novel, and reached by a
   promising rollout) instead of always starting at `reset()`. Siblings then share a prefix, which also sharpens
   credit assignment. This costs the most engineering.

### 3.4 Balancing exploration and exploitation

- **Annealing:** `η_k = η_0 · max(0, 1 − k/K)` or exponential decay, where `k` is the training iteration.
- **Flat-group gating:** apply curiosity only to groups with no reward variance. Curiosity then fills exactly the
  gap where GRPO is blind (H2, H5), and switches itself off as the policy starts succeeding. No schedule to tune.
- **Success-rate adaptive:** `η` scales with `1 − (running train success)`.
- **Uncertainty-aware:** weight novelty by the policy's own confidence at that step. Being curious where you are
  also confident is directed exploration; being curious where you are confused is aimless wandering. Try both
  signs.
- **Caps:** limit the intrinsic return per episode, so the agent can't farm novelty by opening every drawer.
- **Later:** a bandit over `η` values (as in Agent57) if a single schedule isn't enough.

A second probe of the EMPG axis costs nothing: **flip the sign of ζ**. That turns "seek clarity" into "seek
uncertainty", i.e. entropy-seeking exploration. It's a config change only.

---

## 4. Pitfalls to watch for

- **Novelty farming.** Open and close, examine, and toggle all create "new" states. A good state key helps
  (physical state ignores `examine`), and so does a per-episode cap. Watch for rising episode length together with
  flat success.
- **Scale mismatch.** The dense intrinsic signal can dominate a sparse ±1 outcome signal. Keep `η` small (start
  near EMPG's ζ = 0.05) and log the intrinsic share of `|A|`.
- **Noisy TV.** LLM surprisal and observation hashes can reward wording rather than new states. ALFWorld is
  deterministic and templated, which reduces this risk but doesn't remove it.
- **Memory confound (H6).** Always run the memory-in-prompt baseline.
- **Compute.** Curiosity may make episodes longer, and rollouts are the cost bottleneck on this machine. Episode
  caps (`train.max_steps`) bound it, but compare methods at *equal steps*, not just equal iterations.
- **Eval must stay clean.** Curiosity applies during training only. Evaluation stays greedy with the extrinsic
  task metric.
- **Statistics.** One seed is anecdote. Screening can use one seed; claims need ≥3 paired seeds (the same seed
  gives the same task order across arms, as the suite already does).

---

## 5. Testbeds

| Testbed | Why | Status |
|---|---|---|
| **ALFWorld** (current) | Paper benchmark. The six subtasks differ in horizon: Pick/Look are short, Pick2 and Clean/Heat/Cool are long. That gives a built-in "does it help more on long tasks?" test. | ready |
| **TextWorld Coin Collector** (`tw-make tw-coin_collector`) | The classic text-game *exploration* benchmark. It's a maze of rooms with one coin, and the horizon is set by a difficulty level. Count-based exploration was studied on it by Yuan et al. 2018. It's a clean dial for "how long a horizon before GRPO fails and curiosity helps?". Prompts should be much shorter than ALFWorld's, so runs should be faster (to be measured). | `textworld` is installed; needs an `envs/` module |
| **Jericho / Zork** | Very long horizon, the stress test. | `jericho` got installed with alfworld; later |
| **WebShop / ScienceWorld** | The paper's other domain / longer science tasks. | later |

Suggested order: build the method on **ALFWorld**, and use **Coin Collector** to sweep horizon length cheaply.

---

## 6. Metrics (add to `trainer.update` stats)

- **Task:** success per subtask (existing); iterations to reach X% training success (sample efficiency); area
  under the training-success curve.
- **Exploration:** unique states per episode; revisit rate; group coverage (distinct states across the 8
  siblings); P(group has ≥1 success); `flat_group_frac` (existing).
- **Balance:** episode length of *successful* episodes (curiosity shouldn't make solved tasks slower); intrinsic
  share of `|A|`; step entropy (existing).
- **Diagnostics:** correlation between step novelty and final success in base-model rollouts (Phase 0).

---

## 7. Phased plan (sized for the local RTX 5060)

Compute tiers: **smoke** (minutes), **small** (~1.5 h per arm: 10 iterations, 10-turn episodes, 16-game eval) and
**reduced** (~10 h per arm).

**Phase 0: measure before changing anything (≈1–2 h, no training).**
- Add state keys to the env `info` and log them per step.
- Roll out the base model on ~50 training games × 8 samples, then measure revisit rate, group coverage and
  P(≥1 success).
- **Go/no-go gate:** if the base policy rarely revisits and groups are already diverse, curiosity has little room
  to help, and the plan should pivot toward the gating and memory questions.

**Phase 1: implement (code, plus smoke runs only).**
- `empg/curiosity.py`: pure functions, like `advantages.py`. Episodic/group counts from state keys, a
  normalisation function, a schedule function, and a flat-group gate. Unit test them the same way as the EMPG
  equations.
- Extend `AdvantageConfig` with `curiosity: {weight η, key, scope, schedule, gate}`, and add an optional
  `intrinsic` array to `compute_advantages`.
- Keep GRPO and EMPG exactly reproducible: with `η = 0`, results must match today's bit for bit.

**Phase 2: screening (small tier, 1 seed, ≈12 h total, run in batches).**

| Arm | Advantage | Tests |
|---|---|---|
| A | GRPO | baseline |
| B | GRPO + memory-in-prompt | H6 confound |
| C | GRPO + episodic novelty (parsed key) | H1 |
| D | GRPO + group novelty (parsed key) | H3 |
| E | GRPO + episodic novelty (physical oracle) | upper bound on state key |
| F | EMPG | paper method |
| G | EMPG + best of C/D | H4 |
| H | EMPG with ζ sign flipped | seek uncertainty vs clarity |

**Phase 3: balance (small tier).** Take the best curiosity variant and compare constant, annealed and flat-group
gated weights (H5). Also try η ∈ {0.02, 0.05, 0.1}.

**Phase 4: final (reduced tier, ≥3 seeds).** Run GRPO, EMPG, the best curiosity arm and the best EMPG +
curiosity arm. Report the paper-style per-subtask table, the training curves and the exploration metrics.

**Phase 5 (optional):**
- Horizon sweep on Coin Collector.
- LLM surprisal and embedding / RND novelty, the no-privileged-information, LLM-native variants.
- Go-Explore branching.

---

## 8. Code changes (keeping the current modular layout)

| File | Change |
|---|---|
| `empg/envs/alfworld_env.py` | request `facts=True`; put `state_key_physical`, `state_key_knowledge` and `state_key_parsed` into `info` |
| `empg/envs/base.py` | document the optional `state_key` in the `Env` protocol, so future benchmarks can supply one |
| `empg/rollout.py` | store `state_key` (of `s_t+1`) and the observation on each `Step` |
| `empg/curiosity.py` (new) | novelty counts per scope, normalisation, schedules, gating (pure numpy) |
| `empg/advantages.py` | `+ η·ĉ_t` term; `η = 0` means identical to current results |
| `empg/prompts.py` | optional "visited locations" line, for the H6 baseline |
| `empg/trainer.py` | exploration metrics; iteration index passed in for schedules |
| `benchmarks/configs/` | `small.yaml` tier; one small YAML per curiosity arm |
| `tests/test_curiosity.py` | invariants: revisit gets 0 bonus, first visit gets maximum, siblings sharing a state split it, the gate turns off in non-flat groups |

---

## 9. Background reading

- Pathak et al., 2017. *Curiosity-driven Exploration by Self-supervised Prediction* (ICM).
- Burda et al., 2018. *Exploration by Random Network Distillation* (RND); and *Large-Scale Study of
  Curiosity-Driven Learning* (the noisy-TV problem).
- Bellemare et al., 2016. *Unifying Count-Based Exploration and Intrinsic Motivation* (pseudo-counts); Tang et al.,
  2017. *#Exploration* (hash-based counts).
- Badia et al., 2020. *Never Give Up* (episodic × lifelong novelty) and *Agent57* (bandit over exploration rates).
- Savinov et al., 2018. *Episodic Curiosity through Reachability*.
- Ecoffet et al., 2019/2021. *Go-Explore*.
- Yuan et al., 2018. *Counting to Explore and Generalize in Text-based Games* (count-based exploration on Coin
  Collector).
- Wang et al., 2025. *Harnessing Uncertainty* (EMPG). This repo's baseline.

---

## 10. Open questions to decide along the way

- Is the physical-state oracle acceptable in the final method, or only as an upper bound? For a general agent
  claim, the headline result should use the parsed key or a learned key.
- Should curiosity be counted on `s_t+1` (the state the action led to, like EMPG's `f`) or on the action itself
  ("tried a new action here")? The two reward different things when an action fails.
- How to report compute fairly when curiosity changes episode length: per iteration, or per environment step?
- If H6 holds (memory explains most of the gain), should the project become "curiosity + memory", or should the
  memory baseline be adopted and curiosity tested on top of it?
