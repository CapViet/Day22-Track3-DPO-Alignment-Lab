# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Đức Kiên Trung - 2A202600769
**Cohort:** AI20K-2
**Tier đã chạy:** T4
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab **Tesla T4** (15.6 GB, max 14.56 GB usable) |
| CUDA / driver | CUDA capability 7.5 · Toolkit 12.8 · Torch 2.10.0+cu128 · Triton 3.6.0 |
| Stack | Unsloth 2026.5.2 · Transformers 5.5.0 · Xformers 0.0.35 · bf16 = **False** (T4 → fp16) |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit, LoRA r=16 / α=32) |
| SFT dataset slice | `saillab/alpaca-vietnamese-cleaned` · 500 samples · 1 epoch · lr 2e-4 |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab) |

> **Environment note:** the run only succeeded after the runtime loaded **Unsloth 2026.5.2**, whose backward pass ("smartly offload gradients / double buffering") avoids the xformers `memory_efficient_attention_backward` path that has no T4 (sm_75) operator for Qwen's grouped-query attention. An earlier attempt crashed with `NotImplementedError ... GPU has capability (7,5)` purely because a stale Unsloth was still imported — a Restart + run-all fixed it. So the crash was an environment/version issue, not a hard T4 limit.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training steps (NB) | 63 (500 ex × 1 ep, eff. batch 8) | 250 (2000 ex × 1 ep, eff. batch 8) |
| Training time (NB3) | — | _~30–40 min on free T4 (wall-clock, approx.)_ |
| VRAM peak | _~10 GB (approx.)_ | _~13–14 GB (T4 max 14.56 GB)_ |
| Final loss | 1.6332 (SFT) | 0.7255 (DPO) |
| Reward gap (chosen − rejected, end of training) | n/a | **+0.369** |
| End chosen reward (log π/π_ref) | n/a | −0.630 |
| End rejected reward (log π/π_ref) | n/a | −0.999 |

**Data-fit caveat (important):** the NB2 length check showed median chosen length = 400 tokens, P95 = 811, and **only 44.2% of the 2000 preference pairs fit inside `MAX_LEN=512`**. More than half the pairs were truncated, which weakens the preference signal DPO sees (see §3 and §6).

**Tulu 3 reference numbers** (deck §7.2b, context only): +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO on Llama-3-8B-Instruct, 70B-class scale — not expected to replicate at 3B).

---

## 3. Reward curves analysis (≥ 100 words)

> See `submission/screenshots/03-dpo-reward-curves.png` (dual curve + gap).

The reward gap ended **positive at +0.369**, and the NB3 self-check classified the run as "intended" because the *chosen* reward trended **up** over the last steps relative to the first steps. But the more honest reading comes from the absolute values: at end of training the chosen reward is **−0.630** and the rejected reward is **−0.999** — *both are negative*. A negative implicit reward means the policy assigns **lower** probability to that response than the frozen SFT reference does. So DPO did not make the chosen answers more likely than the SFT model; it made *both* less likely, and the gap only grew because the rejected answers were pushed down faster. This is the borderline regime the deck warns about in §3.4: a healthy-looking positive gap can coexist with the chosen probability actually eroding versus the reference. The chosen curve recovering toward −0.630 (rather than collapsing) is what keeps this on the right side of likelihood displacement, but it is a *weak, early-training* signal — consistent with lr 5e-7, a single 250-step epoch, and the 44.2%-fit truncation starving the loss of clean pairs. The takeaway: the gap is the headline, but the separate chosen trajectory tells me the model has barely started moving, not that alignment is "done."

---

## 4. Qualitative comparison (≥ 8 examples)

> See `submission/screenshots/04-side-by-side-table.png` (8 prompts × 2 models).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích Quicksort (5–7 câu) | Coherent VN explanation | Near-identical text | tie |
| 2 | helpfulness | (helpfulness prompt 2) | coherent | near-identical | tie |
| 3 | helpfulness | (helpfulness prompt 3) | coherent | near-identical | tie |
| 4 | helpfulness | (helpfulness prompt 4) | coherent | near-identical | tie |
| 5 | safety | (safety prompt 1) | reasonable refusal/answer | near-identical | tie |
| 6 | safety | (safety prompt 2) | reasonable | near-identical | tie |
| 7 | safety | (safety prompt 3) | reasonable | near-identical | tie |
| 8 | safety | (safety prompt 4) | reasonable | near-identical | tie |

**Win/loss/tie summary:** SFT+DPO 0/8 · SFT-only 0/8 · **tie 8/8** (helpfulness 4 tie, safety 4 tie).

**Judge used:** manual rubric (no `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` set → fallback mode).

**Honest interpretation:** the all-tie result has two compounding causes. (1) No judge API key was set, so the automatic verdict defaulted to "tie." (2) More fundamentally, the SFT and SFT+DPO generations are **nearly identical token-for-token** — which matches the §3 finding that DPO moved the policy only slightly (gap +0.369, both rewards still below the reference). With so little behavioral change, ties are the *correct* call, not a bug. To get a meaningful win-rate I would need a stronger DPO signal first (see §6), then re-run with an actual judge model.

---

## 5. β trade-off

I did **not** run the β-sweep bonus this time. Hypothesis (deck §3.3): β controls how hard DPO is allowed to pull the policy away from the reference. With my **β=0.1** run already producing only a +0.369 gap and both rewards negative, I'd expect:
- **β=0.05** (looser KV leash) → larger reward gap and bigger behavioral change, but higher risk of likelihood displacement / degeneration and reward over-optimization.
- **β=0.5** (tighter leash) → an even smaller gap, outputs almost indistinguishable from SFT — i.e. more ties than I already got.

So I'd predict reward gap to be **monotonically decreasing in β**, with the win-rate sweet spot somewhere below 0.1 *for this truncated data* — but only if the truncation problem is fixed first, otherwise lowering β just amplifies noise from half-truncated pairs.

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _(predict ↑ vs 0.1)_ | _predict > 0_ | _predict longer_ | not run |
| 0.1 (default) | +0.369 | 0/8 (all tie) | ~SFT length | this run |
| 0.5 | _(predict ↓ vs 0.1)_ | _predict 0_ | _≈ SFT_ | not run |

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

The decision I keep coming back to is **staying on the free T4 tier with `MAX_LEN=512`** rather than moving to the BigGPU tier with `MAX_LEN=1024`.

1. **The alternative I considered:** switching the runtime to an L4/A100 (BigGPU config), which would have let me keep full-length preference pairs and run the larger 7B base.
2. **Why I picked T4:** cost ($0) and availability — it's the path every student can reproduce, and the rubric explicitly grades the clarity of my before/after, not absolute speed.
3. **Did the result confirm or surprise me?** It surprised me, and the NB2 diagnostic is what made it visible: **only 44.2% of the 2000 preference pairs fit inside 512 tokens** (median chosen length 400, P95 811). I had assumed 512 was "fine for a 3B lab." In reality, more than half my training pairs were truncated mid-answer, which I now believe is the single biggest reason the reward gap stayed small (+0.369) and every NB4 comparison tied — DPO was learning from chopped-off chosen responses, so the preference signal was muddy.
4. **If I redid it tomorrow:** I would either **filter the preference set to pairs that fit in 512** before training (so every gradient comes from a complete pair) or accept a smaller, cleaner slice at a longer `MAX_LEN`. I'd also set a judge API key so NB4 produces a real win-rate instead of defaulting to ties. The lesson: on a memory-constrained tier, *data that fits* matters more than *data that is plentiful* — I'd spend my next hour on data hygiene, not on more steps.

---

## 7. Benchmark interpretation (≥ 150 words)

> Optional (bonus NB6) — **not run** in this submission. Hypothesis below.

I did not run the NB6 quantitative benchmark (IFEval / GSM8K / MMLU / AlpacaEval-lite) this time, so I have no measured deltas to report. Based on the §3/§4 evidence — a small +0.369 reward gap, both implicit rewards still below the SFT reference, and 8/8 ties in the side-by-side — my hypothesis is that **all four benchmarks would move only marginally**, well within noise, because DPO barely shifted the policy. Specifically: I'd expect **IFEval** to be roughly flat or slightly up (instruction-following is what preference data nudges first), **GSM8K and MATH** flat-to-slightly-down as a mild alignment tax (deck §8.1 — preference tuning rarely improves arithmetic and can erode it), **MMLU** essentially unchanged (factual knowledge lives in the frozen 4-bit base, which LoRA + a 250-step DPO pass shouldn't disturb — so no catastrophic forgetting), and **AlpacaEval-lite** win-rate near 50% / tie, consistent with the NB4 manual result. The honest summary is that this run is a *correctly-wired pipeline with a deliberately weak training signal*; I would not expect it to demonstrate a real alignment improvement until the truncation issue from §6 is fixed and training is extended. Running NB6 would mostly confirm "no regression, no clear gain."

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

Reward gap có thể **dương** (+0.369) trong khi *cả hai* implicit reward đều **âm** — nghĩa là DPO làm câu "chosen" ít khả năng hơn so với reference, chỉ là "rejected" tụt nhanh hơn. "Gap đi lên" không tự động nghĩa là model tốt lên (deck §3.4). Và thủ phạm lớn nhất lại là chi tiết tưởng nhỏ: chỉ 44.2% preference pairs lọt vào `MAX_LEN=512`.
