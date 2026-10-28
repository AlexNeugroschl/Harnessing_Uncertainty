# Practical EMPG implementation

The default command now runs `paper_optimized`: GRPO versus EMPG, one training seed, LoRA
rank 16, two games with eight siblings each, 20 iterations, 50-turn episodes, and all 134
`valid_unseen` evaluation tasks. Outputs use `runs/paper_optimized` and
`results/paper_optimized` directories. The suite deadline is 8.5 hours, including evaluations.
The historical `paper_overnight` tier remains unchanged at 12 iterations. See
[update optimizations](update_optimizations.md) for the new execution settings, measurements,
and how to reuse that completed seed in a compatible replication report.

## Protocol corrections

The [paper, appendix C and E](https://arxiv.org/html/2509.09265v1) and the
[publication-era framework](https://github.com/langfengQ/verl-agent/tree/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd)
are the references. The corrected tiers select:

- Learning rate 1e-6, KL coefficient 0.01, differentiable entropy coefficient 0.001,
  weight decay 0.01, and symmetric PPO clip 0.2. The pinned actor's additional negative-advantage
  dual-clip bound of 3 is enabled; legacy tiers keep their original loss.
- Framework response-row rewards and normalization, with format penalties local to the
  malformed response. EMPG then applies its step entropy modulation and centering.
- Appendix min-max and scaling-denominator epsilon of 1e-8, with no successor bonus at
  episode boundaries. Entropy is computed over the full vocabulary, not sampled tokens.
- Publication-era action projection, two-observation history, a 512-token response cap and
  a 2,048-token prompt limit. An overlong prompt raises an explicit error; no goal/history
  is silently deleted to make it fit.
- Sequential optimizer minibatches of at most 256 response rows. Short batches take fewer
  optimizer steps instead of always splitting into four. Each microbatch contributes its
  share of the minibatch's total response tokens to the loss.
- Sampled evaluation at temperature 0.4 with explicit sampling seed 0. Common evaluation
  randomness makes engine reuse independent of training RNG consumption. The training
  distribution remains temperature 1, top-p 1 and repetition penalty 1.

Remaining compromises are explicit: LoRA instead of updating the entire model, two games
instead of sixteen, and padded length-sorted microbatches instead of the distributed unpadded
actor. Evaluation runs once after training to avoid repeated full-panel rollout costs, instead
of the public script's periodic validation. The overnight iteration budget is smaller.
`paper_lora_fast` retains 150 iterations;
its root-runner seed default is 0, 1, 2. Neither setting is an exact reproduction of the
authors' multi-A100 experiment. Exact token-segment boundary parity remains unverified because
the appendix's segmentation helper was not published.

## Avoiding unnecessary work

Compiled vLLM handles decoding with prefix caching and at most 32 scheduled sequences. Its
context limit is 2,560 tokens, matching the combined prompt and response limits. BF16 autocast
keeps trainable LoRA parameters in float32. Checkpointing stays enabled; exact vocabulary scores
are projected only at response positions and in bounded chunks. GPU work is not replaced with
an approximate entropy objective.

Training and evaluation share one model and engine for corrected tiers. This avoids a second
model load and graph initialization per method. Every changed adapter receives a fresh ID;
the previous adapter is removed. The engine sleeps while HF scores and updates the policy.
Shutdown moves the HF model to CPU, wakes the inference allocator's mappings, and explicitly
shuts down its worker before removing temporary adapter files.

Adapter saves pass `save_embedding_layers=False`: this policy never resizes the vocabulary or
trains base embeddings. That avoids PEFT's repeated remote configuration check during rollout
adapter saves. All adapter weights remain saved; the round-trip regression checks them.
Normal first-use model downloads remain available. No Hub upload is performed by training.

The default excludes curiosity, memory and sign-flip extensions; they remain selectable with
`--arms`. Completed runs are skipped and incompatible configurations are rejected. Historical
tiers retain their settings and fingerprints. The performance profiler measures the configured
path by default; dense reference updates require explicit `--compare-dense` because they cost
extra time and memory.

## Commands

Windows automatically selects `~/venvs/empg-vllm` in WSL for the default fast tier. Inside
WSL activate that environment. `EMPG_WSL_PYTHON` still overrides interpreter selection.

```bash
python run_benchmarks.py --estimate-only
python run_benchmarks.py
python run_benchmarks.py --hours 8

# Longer comparison: inspect cost first. Default here is three seeds.
python run_benchmarks.py --tier paper_lora_fast --estimate-only

# Profile one full-horizon batch on this machine without a dense reference update.
python -m benchmarks.profile --config empg --tier paper_overnight \
  --model-only --turns 50 --samples 16 --out runs/audit/paper_horizon_profile.json
```

For HF-only decoding, use `paper_lora`, or override `policy.backend=hf` in a fresh output
directory. The normal `empg` environment supports that fallback. Do not compare arms using
different backends in one experiment. The timing estimate incorporates measured component
gains and the full-horizon profile but remains an extrapolation; a deadline can leave an arm unfinished.
The optimized default estimate is about 7.5 hours centrally, or 7.1 hours with the
existing zero-shot baseline reused, with an 8.5-hour deadline. It uses the completed
5.1-hour suite and the new shared-rollout update probe; see the optimization notes.
The broad 0.5-1.5x workload range is not a measured timing interval.

## Validation

All 224 unit/CPU integration tests pass. Coverage includes appendix constants, response
minibatches, masked entropy gradients, legacy fingerprints, evaluation RNG isolation, adapter
save round trips without Hub checks, inference cleanup order, the dual-clip loss/gradient, and
requiring final-save completion markers for interrupted corrected runs.

The final real GPU smoke run completed base evaluation and two training iterations followed
by evaluation for both GRPO and EMPG in 430 seconds. It exercised the compiled backend,
adapter refresh, and shared-engine evaluation. The later dual-clip bound is covered separately
by analytic gradient and CPU training tests; clipping was inactive in this smoke run.
All three inference
workers shut down without the prior allocator/forced-kill warnings. Evidence is under ignored
`runs/audit/implementation_final_smoke.log` and `runs/audit/implementation_final_report/`.
The full-horizon probe used 16 trajectories, each reaching all 50 turns. Its rollout took
350.86 seconds including cold startup/transfers, with 232.51 seconds decoding. The 800-response
update took 551.64 seconds including 192.80 seconds scoring. The HF process peaked at 4.70 GiB
reserved (excluding the sleeping inference worker); sampled total physical use during the
update was 4.9-5.3 GiB. Maximum observed prompt length was 1,090 tokens and response length 512.
The loss remained finite and the update completed all four optimizer minibatches.

That probe's engine exceeded the 10-second shutdown grace and was stopped by the bounded
process-manager fallback. Metrics were written successfully; no CUDA allocator error recurred.
This remains an external backend cleanup caveat. The probe predates the final dual-clip flag;
its memory/timing evidence applies to the same model/scoring path, while the new loss bound
is checked independently by analytic and CPU integration tests. Evidence is in
`runs/audit/implementation_horizon_profile.json` and its log. These are implementation checks,
not evidence of a learning improvement. No full overnight benchmark was launched.
