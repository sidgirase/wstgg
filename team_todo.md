# Four-lane work breakdown

> **Window dates in this document are superseded by `todo_by_lane.md` / `todo_tasks.csv`**, which
> rebase the schedule to an Oct 1 start (14 days to the midterm, one task to begin with). Use this
> document for the *content* of each lane and the separation rules; use the other two for *when*.

CS 8803-VLM Fall 2026 · project: perception/reasoning profile → where the LoRA budget goes.
Lanes are sized to be comparable, and each one is a **vertical slice** (code + experiments + a
section of the writing) because the Nov 30–Dec 4 TA interview grades *individual* understanding and
contribution, and insufficient demonstrated contribution can downweight that student across the
whole 50%. Map A/B/C/D to names at the first W5 meeting and put the mapping on the midterm slide —
the midterm rubric asks for implementation ownership explicitly.

## 0a. Separation from the concurrent ICLR submission

One team member has a paper under review on on-policy distillation vs. SFT, built on the same
harness. The de-duplication is structural, not cosmetic, and every lane carries a piece of it:

| Axis | The ICLR submission | This project |
|---|---|---|
| Question | which **supervision source** (teacher vs. filtered teacher data) | which **parameter locus** under a fixed budget |
| Teacher | Qwen3-VL-8B teacher, central to the method | **none — no teacher, no distilled supervision** |
| Tasks | ChartQA, Geometry3K | InfographicVQA, TabMWP, AI2D |
| OOD | CharXiv, ChartQAPro | DocVQA |
| Primary models | Qwen3-VL-2B / 8B | InternVL3-2B, SmolVLM2-2.2B (Qwen3-VL-2B third) |
| Headline analysis | data-efficiency curve, token-level KL | profile-vs-delta correlation, adapter-subspace gradient alignment |

Rules that follow: do not run ChartQA or Geometry3K in this project; do not reproduce the
data-efficiency curve or the termination-rate analysis; do not introduce a teacher model, even as a
convenience for generating oracle text. Inheritance is limited to the harness, the scorer convention,
and the seeding discipline, each cited as an external resource.

## 0. Revision to the earlier scope cut

With four people the fourth lane should buy **sample**, not a second hypothesis. The cut keeps:

- **Dropped:** merger-only placement; auxiliary-perception-supervision (H2) stays deferred as a
  planned ablation.
- **Restored by having a 4th person:** a second model *and* a third trainable task → 6 (task, model)
  pairs instead of 4. Pair count is what the prediction claim rests on, so this is the highest-value
  use of a fourth lane. H2 is a second hypothesis whose unconditional form is already published
  (Wu et al., arXiv:2605.29496) — it buys less.
- Run budget: 2 placements × 1 supervision × 3 seeds = 6 training runs per pair → **36 runs** over
  6 pairs.

## 1. Tonight and this week (whole team)

| When | Item | Owner |
|---|---|---|
| **Sep 16, 11:59 PM ET** | Proposal slides uploaded | whoever presents Sep 17 |
| Sep 17 | 4-minute talk; follow `slides_4min.md` timings, not the full draft | 1 presenter |
| First W5 meeting | Map A–D to names; create the repo fork; agree the claim-discipline rule (4–6 pairs supports "profile and delta move together", never "validated predictor") | all |

## 2. Lane A — Stage A diagnostic (the go/no-go #1 owner)

Owns the instrument: the three inference conditions and the attribution table. Heaviest in W5–W6,
lighter after.

**W5 (Sep 22–26)**
- [ ] InfographicVQA oracle text — **start here, it is the one confirmed zero-work source**: the
      `InfographicVQA` config of `lmms-lab/DocVQA` carries an `ocr` field per image, plus an
      `operation/reasoning` label per question. Write `oracle.py::infovqa_ocr(row)` and keep the
      reasoning label as a stratification column.
- [ ] TabMWP oracle text: the official release provides every table in textual form alongside the
      rendered image. Confirm the HF mirror carries the `table` field; if it is gated, pull from the
      official release.
- [ ] AI2D oracle text: `lmms-lab/ai2d` has only `question / options / answer / image` — the diagram
      annotation graph (text boxes, arrows, blobs and their relations) is in the **original** release.
      Verify the join rate before committing AI2D as a pair; **if the join fails, drop to two tasks**
      rather than substituting model-generated descriptions (that would reintroduce a generative
      confound the design exists to avoid). Budget 2 days; this is the lane's main risk.
- [ ] Three-condition prompt builder in `diagnose.py`: `image_q`, `oracle_q` (no image), `q_only`.
      Reuse the existing chat-template path so prompts are identical apart from content.

**W6 (Sep 29–Oct 3)**
- [ ] Run all three conditions × {InternVL3-2B, SmolVLM2-2.2B} × {InfographicVQA, TabMWP, AI2D},
      500 test items each, greedy decoding. Inference only — no training, and no teacher model.
- [ ] Attribution logic: per item → perception-limited / reasoning-limited / already-solved /
      inconsistent; exclude `q_only`-correct as shortcut and report that mass separately.
- [ ] Produce the 2×2 attribution table per (task, model) and **the variance check**: do the profiles
      differ materially across pairs? This is go/no-go criterion #1.
- [ ] Stratify the InfographicVQA profile by its annotated operation/reasoning type — a readout the
      prior work on this harness does not have.

**W7–W8**
- [ ] Extend the diagnostic to the pairs Lane D adds; hand profiles to Lane B as the predicted
      placement per pair *before* B's runs finish (the prediction must be registered in advance).
- [ ] Own the "Stage A / diagnostic" section of the midterm slides and the report.

**Nov**
- [ ] Re-run the diagnostic on the fine-tuned checkpoints to check whether the profile shifted the way
      the hypothesis predicts; own that analysis section.

**Done when:** a committed table of profiles per pair, with the shortcut mass reported, plus a
one-paragraph verdict on criterion #1.

## 3. Lane B — Stage B placement and training (the compute lane)

Owns everything that moves the LoRA budget. Heaviest W7–W8 and again Oct 20–Nov 12. Lighter writing
load to compensate.

**W5**
- [ ] `vlm_opd/modeling.py` currently freezes everything and matches LoRA only on language-model
      modules; `check_trainable` asserts no vision parameter is trainable. Generalise to a
      `placement` argument: `lm` (default), `vision` (visual attention `qkv`/`proj` and MLP
      `linear_fc1`/`linear_fc2` plus `visual.merger`). Invert the assertion per placement so the test
      still guards the configuration.
- [ ] Parameter-count matcher: pick LoRA rank per placement so trainable counts agree within 10%;
      emit the realised counts into the run config. Add a unit test in `tests/test_modeling.py`.

**W6**
- [ ] Memory: vision-side LoRA under dynamic resolution is the risk. Keep the 768-px long-side cap and
      the 300–600 image-token bounds, enable gradient checkpointing on the visual blocks, record peak
      VRAM per placement. Fallback if it will not fit: merger + last-k vision blocks (document k).
- [ ] Smoke run of each placement on 100 InfographicVQA questions, 1 seed, to prove the path end to
      end. **Never ChartQA or Geometry3K — see §0a.**

**W7 (Oct 7–10, after Fall break)**
- [ ] First real runs: 2 placements × InfographicVQA × 1–2 model families × 1 seed, identical data and
      identical optimizer steps; TabMWP and AI2D join the grid as their oracle builders land. Log
      per-step time and peak memory.
- [ ] PACE job scripts: extend `slurm/point.sbatch` / `sweep.sh` to the placement axis; make the sweep
      resumable (the existing checkpoint-every-N pattern).

**W8 (by Oct 14)**
- [ ] Seeds 2 and 3 on both tasks, both placements — this is what criterion #2 needs.
- [ ] Hand the result files to Lane C the moment each run lands; do not compute your own CIs.

**Oct 20 – Nov 12**
- [ ] Same grid on the pairs Lane D adds; 4th seed on whichever cells are closest to the decision
      boundary.
- [ ] Own the "method / placement" section of the final presentation and report.

**Done when:** every cell in the grid has a committed run config, a realised trainable-parameter
count, a peak-memory number, and a result file.

## 4. Lane C — evaluation, statistics, and the audit packet

Owns every number anyone quotes, and the Nov 17 deliverable. Steady load, spikes at Oct 14 and
Nov 17. **Lands first: nothing else produces trustworthy numbers until the scorer exists.**

**W5 — critical path, finish by Sep 26**
- [ ] Single scorer used by every condition and every run: relaxed-match accuracy (5% numeric
      tolerance, the standard convention for numeric VQA answers) and exact match where that does not
      apply. One code path, in `evaluate.py`; no per-experiment variants.
- [ ] Answer-parse-failure and token-cap-hit metric (fraction of generations that produce a parseable
      answer, and the fraction that hit the token cap) — a known small-VLM failure mode that otherwise
      shows up as an accuracy difference with no cause attached.
- [ ] Freeze the test splits (500 items per task) and commit the id lists.

**W6**
- [ ] Paired bootstrap over questions, pooled across seeds, with the paired delta computed over shared
      seeds only. Reuse the existing `tests/test_stats.py` discipline.
- [ ] Seed orchestration convention: seed controls data order and sampling; test set fixed. Document
      it once, in the README.

**W7–W8**
- [ ] Compute every reported number and the CIs; produce the midterm results table.
- [ ] **Criterion #2:** is the matched-budget placement delta > 3 points with a paired CI excluding
      zero on at least one pair? Report yes/no plainly, including if the answer is no.

**Nov 17 packet (start Nov 3, do not start Nov 15)**
- [ ] Repo link + immutable tag; README covering setup, training, inference, evaluation;
      dependency/environment pins; data and checkpoint instructions.
- [ ] Root `AGENT_USAGE.md` — what AI/agent assistance was used, for what, which files and experiments
      it materially affected, and how the output was validated. **Everyone appends to this weekly;**
      C owns the file's coherence, not its content.
- [ ] External-resource attribution: the harness is reused from a prior personal project of one team
      member — state it plainly. Also the datasets and base models.
- [ ] Team-contribution statement, and the 2–3 page method/claims/experiments/results summary.

**Done when:** the packet is submitted on an immutable tag and a teammate can reproduce one headline
number from a clean checkout.

## 5. Lane D — pair expansion, literature watch, rebuttal

Owns the sample size. This is the lane that makes the prediction claim possible rather than anecdotal.

**W5**
- [ ] Write the `_convert_*` adapters for all three new tasks (InfographicVQA, TabMWP, AI2D) and
      register them in the `SOURCES` dict; sample train/test at matched sizes and image cap. The old
      ChartQA and Geometry3K adapters stay in the repo but are **not used** — they belong to the
      concurrent ICLR submission and keeping them out of this project is what makes the two separable.
- [ ] Confirm each new task has a train split (DocVQA is held out for distribution shift, not trained).

**W6**
- [ ] Integrate both primary models (InternVL3-2B, SmolVLM2-2.2B). Expect three friction points per
      family: a different chat template, different vision-module names for the LoRA target regex, and
      a different processor / image-token accounting. Prove each with its zero-shot number before
      anything is trained.
- [ ] Build the per-family module map that Lane B's placement code consumes; one unit test per family.
- [ ] Coordinate with A so the diagnostic runs on every pair.

**W7–W8**
- [ ] Zero-shot baselines for every new pair; hand to C for the results table.
- [ ] Own the "related work" slide at the midterm, kept current.

**Ongoing, 30 min/week**
- [ ] Scoop watch. This area published four close papers in nine months (MathVerse, CapGeo-Bench,
      CoVFT, Wu et al.). Standing arXiv query on perception/reasoning decoupling + VLM PEFT placement;
      report anything that lands to the team within the week. If something scoops the conditional
      claim, we want to know in October, not in November.

**Nov 20–24 — the rebuttal (10% of the course)**
- [ ] Draft the response to the AI-assisted audit: per finding, agree / partly agree / disagree, with
      evidence — code pointers, a run, a citation, or a check — and state the effect on our claims.
      Expect attacks on pair count, on whether rank-matching really equalises compute, and on seeds;
      the pre-registered go/no-go criteria are the strongest asset here.

**Done when:** 6 pairs exist with zero-shot baselines, and the rebuttal is submitted by Nov 24.

## 6. Dependencies and the critical path

- **C's scorer (W5) gates everything.** No one quotes a number before it lands.
- **A → B:** the predicted placement per pair must be registered *before* B's runs for that pair
  finish, or the prediction is post hoc. Commit the prediction to the repo with a timestamp.
- **D → A and D → B:** the new task and model must land in W6 or they miss the midterm grid.
- **B → C:** result files flow to C; B never computes its own CIs.
- Critical path: `C scorer (Sep 26) → B smoke (Oct 3) → B runs (Oct 10) → C CIs (Oct 14) → midterm`.

## 7. Shared obligations

- [ ] Weekly 30-minute sync; one person writes three lines of minutes into `docs/resume_log.md`.
- [ ] Everyone appends to `AGENT_USAGE.md` the same week they use assistance, not in November.
- [ ] **W13 mutual walkthrough (before Nov 30).** Each person walks their lane's code to the other
      three for 15 minutes. The TA interview asks each student about the system, implementation,
      experiments, evaluation, audit and contributions — owning one lane is necessary but not
      sufficient.
- [ ] Final presentation (Dec 1/3/8): A presents the diagnostic, B the placement method, C the
      evidence and statistics, D the related work and limitations. Final report sections follow the
      same split; C integrates.

## 8. If a go/no-go fails on Oct 15

Criterion #1 fails (profiles identical across pairs): there is nothing to predict. Pivot to the
measurement paper — "what small VLMs actually get wrong on chart and diagram tasks, and how much of
it is perceptual" — which Lane A's work already supports, and reassign B's compute to the OOD sweep.

Criterion #2 fails (delta inside noise): report the null as a small-model confirmation of Zhou et al.
(2026), keep the diagnostic as the contribution, and spend November on depth (more seeds, the
termination-rate analysis) rather than more cells. Both outcomes are reportable under the final-report
rubric, which asks for limitations and threats to validity explicitly.
