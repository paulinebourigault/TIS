#!/usr/bin/env python3
"""A live, history-dependent review loop with checkpoint restarts (Appendix H).

Configurations: configs/rebuttal_live_loop.json (study 1), rebuttal_rare_live.json
(study 2) and rebuttal_late_live.json (study 3), selected with LIVE_LOOP_CONFIG. Every transition is a fresh model call that
writes brief reasoning and sees the whole conversation. Complete runs make
H = 3 calls (solve, two reviews); the return is the Brier utility of the final
answer. Checkpoint designs restart from conversation prefixes stored during
their own pilot runs. Estimates use the paper's categorical estimator on
stage-state groups; errors are measured against a 10,000-run live reference.

Usage (add --stub to any model step to test without a GPU; --shard k/K splits
reference questions or replications across K GPUs, one process per GPU):
  python scripts/rebuttal_live_loop.py smoke [--stub]      # format and throughput only
  python scripts/rebuttal_live_loop.py decide GPUS         # scale rule
  python scripts/rebuttal_live_loop.py screen [--stub]
  python scripts/rebuttal_live_loop.py reference [--stub] [--shard k/K]
  python scripts/rebuttal_live_loop.py run [--stub] [--shard k/K] [--questions a:b]
(--questions a:b runs retained questions a..b-1 of the screening order with the
seeds they have in a run of all retained questions.)
  python scripts/rebuttal_live_loop.py analyze
Writes results/rebuttal_live_loop/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.protocol import (  # noqa: E402
    ANSWER_LABELS, CONFIDENCE_LABELS, action_for_state, brier_utility, format_item, output_coordinates,
    output_index,
)
from tis.llm_study.runner import _empirical_cvar  # noqa: E402
from tis.model import FiniteHorizonModel, Kernel, model_from_counts  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402

CONFIG = json.loads((ROOT / os.environ.get("LIVE_LOOP_CONFIG", "configs/rebuttal_live_loop.json")).read_text())
OUT = Path(os.environ.get("LIVE_LOOP_OUT", ROOT / CONFIG["output_directory"]))  # override for stub tests
H = int(CONFIG["horizon"])
ALPHA = float(CONFIG["alpha"])
SEED = int(CONFIG["master_seed"])
PROMPTS = CONFIG["prompts"]
POLICY = CONFIG["policy"]
FINAL = re.compile(r"Final: ([A-J])([1-9])\s*$")
STATES = 91  # 0 = before the first call, 1..90 = latest (answer, confidence)
DESIGNS = tuple(d for d in ("ckpt_anchored", "ckpt_occ_plus_mean", "ckpt_tis", "ckpt_uniform") if d in CONFIG["designs"])
IN_WAVES = CONFIG["reference"].get("mode") == "in_waves"  # study 2 and 3: reference runs share the design waves
LABELS = {"answer_labels": "".join(ANSWER_LABELS), "confidence_labels": "".join(CONFIDENCE_LABELS)}


# ----------------------------------------------------------------------------- prompts and parsing
def first_turn(item) -> list[dict]:
    return [{"role": "system", "content": PROMPTS["system"]},
            {"role": "user", "content": PROMPTS["solve_template"].format(item=format_item(item), **LABELS)}]


def review_turn(state: int) -> dict:
    """Instruction appended after a response that ended in `state` (1..90)."""
    action = action_for_state(state, policy=POLICY)
    answer, confidence = output_coordinates(state - 1)
    prior = PROMPTS["prior_template"].format(answer=ANSWER_LABELS[answer], confidence=CONFIDENCE_LABELS[confidence],
                                             confidence_decimal=f"0.{confidence + 1}")
    return {"role": "user", "content": PROMPTS[f"{action}_template"].format(prior=prior, **LABELS)}


def parse_state(text: str) -> int:
    match = FINAL.search(text.strip())
    if match is None:
        return -1
    return output_index(ANSWER_LABELS.index(match.group(1)), CONFIDENCE_LABELS.index(match.group(2))) + 1


def utilities(item) -> np.ndarray:
    """u[s] for s = 1..90 (index 0 unused)."""
    out = np.zeros(STATES)
    for s in range(1, STATES):
        answer, confidence = output_coordinates(s - 1)
        out[s] = float(brier_utility(answer, confidence, int(item["correct_index"])))
    return out


# ----------------------------------------------------------------------------- model backends
class VLLMBackend:
    def __init__(self):
        from vllm import LLM, SamplingParams
        model = CONFIG["model"]
        dec = CONFIG["decoder"]
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model["identifier"], revision=model["revision"], tokenizer_revision=model["revision"],
                       dtype=model["dtype"], max_model_len=int(model["max_model_len"]),
                       gpu_memory_utilization=float(model["gpu_memory_utilization"]), enable_prefix_caching=True,
                       max_num_seqs=int(model["max_num_seqs"]), guided_decoding_backend=dec["guided_backend"], seed=0)
        try:
            from vllm.sampling_params import GuidedDecodingParams
            self.label = {"guided_decoding": GuidedDecodingParams(regex=dec["label_regex"])}
        except ImportError:  # newer vLLM
            from vllm.sampling_params import StructuredOutputsParams
            self.label = {"structured_outputs": StructuredOutputsParams(regex=dec["label_regex"])}
        self.decoder = dec
        self.calls = self.prompt_tokens = self.output_tokens = 0

    def generate(self, conversations: list[list[dict]], seeds: list[int]) -> list[str]:
        """Two-stage decoding in chunks, so each chunk's prompts are still cached when its labels are decoded."""
        chunk = int(CONFIG["model"]["chunk"])
        texts = []
        for start in range(0, len(conversations), chunk):
            texts += self._generate(conversations[start:start + chunk], seeds[start:start + chunk])
        return texts

    def _generate(self, conversations: list[list[dict]], seeds: list[int]) -> list[str]:
        """Two-stage decoding: free reasoning (stopped at "Final:"), then a constrained answer-confidence label."""
        if not conversations:
            return []
        dec = self.decoder
        temperature, top_p = float(dec["temperature"]), float(dec["top_p"])
        first = [self.SamplingParams(temperature=temperature, top_p=top_p, max_tokens=int(dec["reasoning_max_tokens"]),
                                     stop=list(dec["reasoning_stop"]), seed=int(s)) for s in seeds]
        t0 = time.perf_counter()
        out1 = self.llm.chat(conversations, sampling_params=first, use_tqdm=False)
        t1 = time.perf_counter()
        reasoning = [o.outputs[0].text.rstrip() for o in out1]
        partial = [conv + [{"role": "assistant", "content": r + "\nFinal: "}] for conv, r in zip(conversations, reasoning)]
        second = [self.SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2, seed=int(s) + 1, **self.label)
                  for s in seeds]
        out2 = self.llm.chat(partial, sampling_params=second, use_tqdm=False, add_generation_prompt=False,
                             continue_final_message=True)
        print(f"[generate] {len(conversations)} calls: reasoning {t1 - t0:.1f}s, label {time.perf_counter() - t1:.1f}s, "
              f"mean prompt {np.mean([len(o.prompt_token_ids) for o in out1]):.0f} tok, "
              f"mean reasoning {np.mean([len(o.outputs[0].token_ids) for o in out1]):.0f} tok", file=sys.stderr, flush=True)
        self.calls += len(out1)
        self.prompt_tokens += sum(len(o.prompt_token_ids) for o in out1)
        self.output_tokens += sum(len(o.outputs[0].token_ids) for o in out1) + sum(len(o.outputs[0].token_ids) for o in out2)
        return [r + "\nFinal: " + o.outputs[0].text.strip() for r, o in zip(reasoning, out2)]


class StubBackend:
    """History-dependent stand-in for local tests: answers drift from the previous answer."""

    def __init__(self):
        self.calls = self.prompt_tokens = self.output_tokens = 0

    def generate(self, conversations, seeds):
        texts = []
        for conv, seed in zip(conversations, seeds):
            rng = np.random.default_rng(int(seed))
            item_key = int(hashlib.sha256(conv[1]["content"].encode()).hexdigest()[:8], 16)
            previous = [parse_state(m["content"]) for m in conv if m["role"] == "assistant"]
            favourite = item_key % 10
            if previous and rng.random() < 0.7:
                answer = output_coordinates(previous[-1] - 1)[0]  # keep the last answer (history dependence)
            elif previous and len(previous) > 1 and rng.random() < 0.5:
                answer = output_coordinates(previous[0] - 1)[0]   # revert to the first answer
            else:
                answer = favourite if rng.random() < 0.6 else int(rng.integers(10))
            confidence = int(rng.integers(9))
            texts.append(f"Reasoning: stub reasoning.\nFinal: {ANSWER_LABELS[answer]}{CONFIDENCE_LABELS[confidence]}")
            self.calls += 1
        return texts


def backend(stub: bool):
    return StubBackend() if stub else VLLMBackend()


# ----------------------------------------------------------------------------- simulation primitives
def simulate(llm, specs, n_jobs, keep_pools=False):
    """Run conversations to the end of the loop, batching every live conversation per call.

    specs: (job, item, prefix, state, h, rng). prefix None starts a fresh run at the root (state 0, h = H);
    otherwise prefix ends with an assistant turn at `state`, with h calls remaining.
    Returns per job: transitions ((h, s), s'), final states, prefix pools by (h, s), invalid responses, calls.
    """
    results = [{"returns": [], "transitions": [], "pools": defaultdict(list), "invalid": 0, "calls": 0}
               for _ in range(n_jobs)]
    convs, owner, states, left, rngs = [], [], [], [], []
    for job, item, prefix, state, h, rng in specs:
        convs.append(first_turn(item) if prefix is None else prefix + [review_turn(state)])
        owner.append(job)
        states.append(state)
        left.append(h)
        rngs.append(rng)
    while convs:
        texts = llm.generate(convs, [int(r.integers(0, 2 ** 31 - 1)) for r in rngs])
        nxt = ([], [], [], [], [])
        for i, text in enumerate(texts):
            res = results[owner[i]]
            res["calls"] += 1
            new = parse_state(text)
            if new < 0:
                res["invalid"] += 1
                res["transitions"].append(((left[i], states[i]), None))
                continue
            res["transitions"].append(((left[i], states[i]), new))
            if left[i] == 1:
                res["returns"].append(new)
                continue
            conv = convs[i] + [{"role": "assistant", "content": text}]
            if keep_pools:
                res["pools"][(left[i] - 1, new)].append(conv)
            for lst, value in zip(nxt, (conv + [review_turn(new)], owner[i], new, left[i] - 1, rngs[i])):
                lst.append(value)
        convs, owner, states, left, rngs = nxt
    return results


def complete_runs(llm, jobs, keep_pools=False):
    """jobs: list of (item, runs, rng); every run starts at the root."""
    specs = [(j, item, None, 0, H, rng) for j, (item, runs, rng) in enumerate(jobs) for _ in range(runs)]
    return simulate(llm, specs, len(jobs), keep_pools)


def allocate_starts(weights: np.ndarray, costs: np.ndarray, total: int, minimum: int) -> np.ndarray:
    """Restart counts proportional to `weights`, each costing `costs` calls, spending at most `total` calls."""
    base = np.full(weights.size, minimum, dtype=int)
    remaining = total - int((base * costs).sum())
    if remaining < 0:
        raise ValueError("budget below the minimum restarts")
    raw = remaining * weights / float((weights * costs).sum())
    n = np.floor(raw).astype(int)
    left = remaining - int((n * costs).sum())
    for i in np.argsort(-(raw - n), kind="stable"):
        if costs[i] <= left:
            n[i] += 1
            left -= int(costs[i])
    return base + n


# ----------------------------------------------------------------------------- estimator
def skeleton(item) -> FiniteHorizonModel:
    u = utilities(item)
    grid = np.unique(np.concatenate([[0.0, float(H)], u[1:]]))
    policy = np.zeros((H + 1, STATES, 1))
    policy[1:, :, 0] = 1.0
    nexts = np.arange(1, STATES)
    kernels = {}
    for h in range(1, H + 1):
        rewards = u[1:] if h == 1 else np.zeros(STATES - 1)
        for s in range(STATES):
            kernels[(h, s)] = Kernel(rewards, nexts, np.full(STATES - 1, 1.0 / (STATES - 1)))
    groups = ((H, 0),) + tuple((h, s) for h in range(H - 1, 0, -1) for s in range(1, STATES))
    return FiniteHorizonModel(name="live_review_loop", horizon=H, initial_state=0, grid=grid, policy=policy,
                              kernels=kernels, query_groups=groups, group_mode="untied_layer_state")


def counts_from(model, transitions):
    counts = {g: np.zeros(STATES - 1, dtype=int) for g in model.query_groups}
    for group, new in transitions:
        if new is not None and group in counts:
            counts[group][new - 1] += 1
    return counts


def fitted_cvar(model, counts) -> float:
    filled = {g: (c if c.sum() > 0 else np.ones_like(c)) for g, c in counts.items()}
    return float(solve_stop_loss(model_from_counts(model, filled), ALPHA).cvar)


# ----------------------------------------------------------------------------- steps
def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, *[int(k) for k in key]]))


def _items(path):
    return [json.loads(line) for line in (ROOT / path).open()]


def _candidate_pool() -> dict:
    sc = CONFIG["screening"]
    sources = sc.get("prefilter", {}).get("sources", []) + sc["sources"]
    return {it["item_id"]: it for src in sources for it in _items(src)}


def _candidates() -> list:
    """Screening candidates in order. Study 3 lists its pre-filtered questions first, then the fresh panels."""
    sc = CONFIG["screening"]
    excluded = set(sc.get("exclude_item_ids", []))
    first = []
    if "prefilter" in sc:
        pool = _candidate_pool()
        first = [pool[i] for i in sc["prefilter"]["item_ids"]]
    ordered = [it for it in first + [it for src in sc["sources"] for it in _items(src)] if it["item_id"] not in excluded]
    if len({it["item_id"] for it in ordered}) != len(ordered):
        raise SystemExit("duplicate screening candidates")
    return ordered


def smoke(stub: bool) -> None:
    llm = backend(stub)
    item = _items(CONFIG["screening"]["sources"][0])[0]
    t0 = time.perf_counter()
    runs = int(CONFIG["scale_rule"]["smoke_runs"])
    res = complete_runs(llm, [(item, runs, _rng(9, 0))])[0]
    seconds = time.perf_counter() - t0
    u = utilities(item)
    finals = [s for s in res["returns"] if s is not None]
    print(f"{runs} runs ({llm.calls} calls) in {seconds:.1f}s = {llm.calls / seconds:.1f} calls/s; "
          f"invalid {res['invalid']}; prompt tokens {llm.prompt_tokens}, output tokens {llm.output_tokens}")
    OUT.mkdir(parents=True, exist_ok=True)
    if not stub:
        texts = llm.generate([first_turn(item)] * 3, [1, 2, 3])
        print("sample responses:", *texts, sep="\n---\n")
    (OUT / "smoke.json").write_text(json.dumps({"calls": llm.calls, "seconds": seconds,
                                                "calls_per_second": llm.calls / seconds, "invalid": res["invalid"],
                                                "prompt_tokens": llm.prompt_tokens, "output_tokens": llm.output_tokens,
                                                "mean_final_utility": float(np.mean([u[s] for s in finals]))},
                                               indent=1) + "\n")


def projected_calls(questions: int, replications: int) -> float:
    # rollouts + pilot + coupled restarts (the union over designs is about 1.4 times one design's restarts)
    per_rep = sum(n + 3 * (n // 9) + 1.4 * (n - 3 * (n // 9)) for n in CONFIG["budgets_calls_per_question"])
    sc = CONFIG["screening"]
    return (questions * replications * per_rep + questions * H * int(CONFIG["reference"]["runs"])
            + len(sc["sources"]) * 50 * H * int(sc["runs"]))


def decide(gpus: int) -> None:
    # Throughput measured on the first reference question, whose prompts carry
    # full histories; the first-turn smoke test overstates it.
    smoke_report = json.loads((OUT / "throughput.json").read_text())
    rule = CONFIG["scale_rule"]
    options = rule["options_in_order"]
    table = [{**o, "projected_hours": projected_calls(o["questions"], o["replications"])
              / (smoke_report["calls_per_second"] * gpus) / 3600} for o in options]
    fits = [o for o, row in zip(options, table) if row["projected_hours"] <= rule["wall_clock_limit_hours"]]
    chosen = fits[0] if fits else options[-1]
    (OUT / "scale.json").write_text(json.dumps({"gpus": gpus, "options": table, "chosen": chosen}, indent=1) + "\n")
    print(json.dumps({"options": table, "chosen": chosen}, indent=1))


def _scale():
    if "scale" in CONFIG:  # fixed in the config (studies 2 and 3)
        return CONFIG["scale"]
    return json.loads((OUT / "scale.json").read_text())["chosen"]


def _screen_keep(sc, it, finals, firsts=None) -> tuple[bool, dict]:
    wrong = float(np.mean([output_coordinates(s - 1)[0] != int(it["correct_index"]) for s in finals]))
    keep = sc["error_rate_band"][0] <= wrong <= sc["error_rate_band"][1]
    extra = {"error_rate": wrong}
    if firsts is not None:  # study 3: errors must be created after the first answer
        first_wrong = float(np.mean([output_coordinates(s - 1)[0] != int(it["correct_index"]) for s in firsts]))
        extra["first_call_error_rate"] = first_wrong
        if "first_call_error_at_most" in sc:
            keep = keep and first_wrong <= sc["first_call_error_at_most"]
    if "minimum_utility_share_below" in sc:  # CVaR must not be the minimum utility
        u = utilities(it)
        low = float(np.mean([u[s] <= u[1:].min() + 1e-12 for s in finals]))
        keep = keep and low < sc["minimum_utility_share_below"]
        extra["minimum_utility_share"] = low
    return keep, extra


def screen_shard(stub: bool, shard, positions=None) -> None:
    """Screen every candidate, batches split across GPUs; `retain` applies the rule in panel order."""
    llm = backend(stub)
    sc = CONFIG["screening"]
    candidates = _candidates()
    first, last = positions or (0, len(candidates))  # candidates added later are screened by position
    batch, report = 20, []
    for b, start in enumerate(range(first, min(last, len(candidates)), batch)):
        if b % shard[1] != shard[0]:
            continue
        chunk = candidates[start:min(start + batch, last)]
        res = complete_runs(llm, [(it, int(sc["runs"]), _rng(0, start + i)) for i, it in enumerate(chunk)])
        for i, (it, r) in enumerate(zip(chunk, res)):
            firsts = [new for key, new in r["transitions"] if key == (H, 0) and new is not None]
            keep, extra = _screen_keep(sc, it, [s for s in r["returns"] if s is not None],
                                       firsts if "first_call_error_at_most" in sc else None)
            report.append({"position": start + i, "item_id": it["item_id"], "subject": it["subject"], **extra,
                           "invalid": r["invalid"], "qualifies": bool(keep)})
        print(f"screened batch {b}: {sum(x['qualifies'] for x in report)} qualifying so far", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_p{first}-{last}_" if positions else ""
    (OUT / f"screening_shard{tag}{shard[0]}of{shard[1]}.json").write_text(
        json.dumps({"candidates_screened": report, "calls": llm.calls}, indent=1) + "\n")


def retain() -> None:
    sc = CONFIG["screening"]
    shards = [json.loads(p.read_text()) for p in sorted(OUT.glob("screening_shard*.json"))]
    report = sorted({c["position"]: c for sh in shards for c in sh["candidates_screened"]}.values(),
                    key=lambda c: c["position"])
    retained = [c["item_id"] for c in report if c["qualifies"]][:int(sc["retain"])]
    for c in report:
        c["retained"] = c["item_id"] in retained
    (OUT / "screening.json").write_text(json.dumps({"retained_item_ids": retained, "candidates_screened": report,
                                                    "calls": sum(sh["calls"] for sh in shards)}, indent=1) + "\n")
    print(f"retained {len(retained)} of {len(report)} screened ({sum(c['qualifies'] for c in report)} qualify)")


def screen(stub: bool) -> None:
    llm = backend(stub)
    sc = CONFIG["screening"]
    excluded = set(sc.get("exclude_item_ids", []))
    candidates = [it for src in sc["sources"] for it in _items(src) if it["item_id"] not in excluded]
    retained, report = [], []
    batch = 20
    for start in range(0, len(candidates), batch):
        chunk = candidates[start:start + batch]
        res = complete_runs(llm, [(it, int(sc["runs"]), _rng(0, start + i)) for i, it in enumerate(chunk)])
        for it, r in zip(chunk, res):
            finals = [s for s in r["returns"] if s is not None]
            wrong = np.mean([output_coordinates(s - 1)[0] != int(it["correct_index"]) for s in finals])
            keep = sc["error_rate_band"][0] <= wrong <= sc["error_rate_band"][1]
            extra = {}
            if "minimum_utility_share_below" in sc:  # CVaR must not be the minimum utility
                u = utilities(it)
                low = float(np.mean([u[s] <= u[1:].min() + 1e-12 for s in finals]))
                keep = keep and low < sc["minimum_utility_share_below"]
                extra = {"minimum_utility_share": low}
            report.append({"item_id": it["item_id"], "subject": it["subject"], "error_rate": float(wrong), **extra,
                           "invalid": r["invalid"], "retained": bool(keep and len(retained) < sc["retain"])})
            if keep and len(retained) < sc["retain"]:
                retained.append(it)
        if len(retained) >= sc["retain"]:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screening.json").write_text(json.dumps({"retained_item_ids": [it["item_id"] for it in retained],
                                                    "candidates_screened": report, "calls": llm.calls}, indent=1) + "\n")
    print(f"retained {len(retained)} of {len(report)} screened; calls {llm.calls}")


def retained_items(every=False):
    screening = json.loads((OUT / "screening.json").read_text())
    pool = _candidate_pool()
    items = [pool[i] for i in screening["retained_item_ids"]]
    scaled = "scale" in CONFIG or (OUT / "scale.json").exists()
    return items[:int(_scale()["questions"])] if scaled and not every else items


def reference(stub: bool, shard=(0, 1), questions=None) -> None:
    llm = backend(stub)
    ref = CONFIG["reference"]
    out = {}
    started = time.perf_counter()
    indexed = list(enumerate(retained_items()))
    name = f"reference_shard{shard[0]}of{shard[1]}.json"
    if questions:  # screening indices a..b-1 with the seeds they have in a full run
        indexed = list(enumerate(retained_items(every=True)))[questions[0]:questions[1]]
        name = f"reference_q{questions[0]}-{questions[1]}_{shard[0]}of{shard[1]}.json"
    for i, (q, item) in enumerate(indexed):
        if i % shard[1] != shard[0]:
            continue
        res = complete_runs(llm, [(item, int(ref["runs"]), _rng(1, q))])[0]
        u = utilities(item)
        values = np.array([u[s] for s in res["returns"] if s is not None])
        rng = _rng(1, q, 1)
        boot = [_empirical_cvar(values[rng.integers(0, values.size, values.size)], ALPHA)
                for _ in range(int(ref["bootstrap_resamples"]))]
        out[item["item_id"]] = {"cvar": _empirical_cvar(values, ALPHA), "bootstrap_variance": float(np.var(boot, ddof=1)),
                                "runs": int(values.size), "invalid": res["invalid"],
                                "error_rate": float(np.mean([output_coordinates(s - 1)[0] != int(item["correct_index"])
                                                             for s in res["returns"] if s is not None]))}
        if not (OUT / "throughput.json").exists():
            (OUT / "throughput.json").write_text(json.dumps({"calls": llm.calls, "seconds": time.perf_counter() - started,
                                                             "calls_per_second": llm.calls / (time.perf_counter() - started),
                                                             "source": "first reference question (full histories)"}) + "\n")
        print(f"reference q{q} item {item['item_id']}: CVaR {out[item['item_id']]['cvar']:.4f} "
              f"(sd {np.sqrt(out[item['item_id']]['bootstrap_variance']):.4f})")
    (OUT / name).write_text(json.dumps({"items": out, "calls": llm.calls}, indent=1) + "\n")


def design_wave(llm, indexed, budget, reps):
    """All designs for the given replications of the (screening index, question) pairs at one budget."""
    exploration = budget ** (-float(CONFIG["checkpoint"]["exploration_exponent"]))
    minimum = int(CONFIG["checkpoint"]["minimum_restarts_per_visited_group"])
    jobs = [(q, item, rep) for rep in reps for q, item in indexed]
    models = {q: skeleton(item) for q, item in indexed}
    u = {q: utilities(item) for q, item in indexed}
    rows = []
    # Complete runs: empirical CVaR and the fitted-model estimator on the same transitions.
    ref_runs = int(CONFIG["reference"]["runs_per_replication"]) if IN_WAVES else 0
    full_jobs = []
    for q, item, rep in jobs:
        full_jobs.append((item, budget // H, _rng(2, q, budget, rep)))
        if ref_runs:  # reference runs share the generate calls of this replication's complete rollouts
            full_jobs.append((item, ref_runs, _rng(5, q, budget, rep)))
    finished = complete_runs(llm, full_jobs)
    full, refs = (finished[::2], finished[1::2]) if ref_runs else (finished, [])
    # Checkpoint pilot runs: their conversations form the restart pool; their transitions are kept.
    pilots = complete_runs(llm, [(item, budget // (3 * H), _rng(3, q, budget, rep)) for q, item, rep in jobs],
                           keep_pools=True)
    specs, plans = [], []
    for (q, item, rep), pilot in zip(jobs, pilots):
        model = models[q]
        pilot_counts = counts_from(model, pilot["transitions"])
        visited = np.flatnonzero([pilot_counts[g].sum() > 0 for g in model.query_groups])
        filled = {g: (c if c.sum() > 0 else np.ones_like(c)) for g, c in pilot_counts.items()}
        bundle = compute_influences(model_from_counts(model, filled), ALPHA)
        occ = _occupancy_scores(model, bundle.occupancy)
        tail, mean = np.asarray(bundle.tail_scales), np.asarray(bundle.mean_scales)
        costs = np.array([model.query_groups[g][0] for g in visited])  # a restart at (h, s) runs h calls

        def floored(score):
            return regularized_weights(score[visited], exploration)

        weights = {"ckpt_anchored": regularized_weights(_anchored_score(tail[visited], occ[visited]), exploration),
                   "ckpt_occ_plus_mean": 0.5 * floored(mean) + 0.5 * floored(occ),
                   "ckpt_tis": floored(tail),
                   "ckpt_uniform": np.full(visited.size, 1.0 / visited.size)}
        weights = {d: w for d, w in weights.items() if d in DESIGNS}
        alloc = {}
        for design, w in weights.items():
            n = np.zeros(model.group_count, dtype=int)
            n[visited] = allocate_starts(w, costs, budget - pilot["calls"], minimum)
            alloc[design] = n
        maximum = np.max(np.vstack(list(alloc.values())), axis=0)
        rng = _rng(4, q, budget, rep)
        streams = {}
        for gi in np.flatnonzero(maximum):
            h, s = model.query_groups[gi]
            streams[gi] = []
            for _ in range(int(maximum[gi])):
                prefix = None if (h, s) == (H, 0) else pilot["pools"][(h, s)][int(rng.integers(len(pilot["pools"][(h, s)])))]
                streams[gi].append(len(specs))
                specs.append((len(specs), item, prefix, s, h, np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))))
        plans.append({"alloc": alloc, "streams": streams, "visited": int(visited.size)})
    restart_results = simulate(llm, specs, len(specs))
    for (q, item, rep), run, pilot, plan in zip(jobs, full, pilots, plans):
        model = models[q]
        finals = np.array([u[q][s] for s in run["returns"]])
        base = {"item_id": item["item_id"], "budget": budget, "replication": rep}
        rows.append({**base, "design": "complete_rollout", "estimate": _empirical_cvar(finals, ALPHA),
                     "calls": run["calls"], "visited_groups": "", "invalid": run["invalid"]})
        rows.append({**base, "design": "rollout_model", "estimate": fitted_cvar(model, counts_from(model, run["transitions"])),
                     "calls": run["calls"], "visited_groups": "", "invalid": run["invalid"]})
        for design, n in plan["alloc"].items():
            used = [restart_results[k] for gi, ks in plan["streams"].items() for k in ks[:int(n[gi])]]
            transitions = pilot["transitions"] + [t for r in used for t in r["transitions"]]
            rows.append({**base, "design": design, "estimate": fitted_cvar(model, counts_from(model, transitions)),
                         "calls": pilot["calls"] + sum(r["calls"] for r in used), "visited_groups": plan["visited"],
                         "invalid": pilot["invalid"] + sum(r["invalid"] for r in used)})
    ref_rows = []
    for (q, item, rep), ref in zip(jobs, refs):
        counts = np.bincount([s - 1 for s in ref["returns"] if s is not None], minlength=STATES - 1)
        ref_rows.append({"item_id": item["item_id"], "budget": budget, "replication": rep, "runs": int(counts.sum()),
                         "invalid": ref["invalid"], "calls": ref["calls"], "final_counts": " ".join(map(str, counts))})
    return rows, ref_rows


def run(stub: bool, replications: int | None, shard=(0, 1), questions=None) -> None:
    llm = backend(stub)
    indexed = list(enumerate(retained_items()))
    path = OUT / f"raw_shard{shard[0]}of{shard[1]}.csv"
    if questions:  # screening indices a..b-1, keeping the seeds they have among all retained questions
        indexed = list(enumerate(retained_items(every=True)))[questions[0]:questions[1]]
        path = OUT / f"raw_shard_q{questions[0]}-{questions[1]}_{shard[0]}of{shard[1]}.csv"
    reps_total = replications or int(_scale()["replications"])
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(int(r["budget"]), int(r["replication"])) for r in csv.DictReader(path.open())}
    wave = int(CONFIG.get("wave_replications_per_process", 5)) * shard[1]
    for budget in CONFIG["budgets_calls_per_question"]:
        for start in range(0, reps_total, wave):
            reps = [r for r in range(start, min(start + wave, reps_total))
                    if (budget, r) not in done and r % shard[1] == shard[0]]
            if not reps:
                continue
            t0 = time.perf_counter()
            rows, ref_rows = design_wave(llm, indexed, budget, reps)
            if ref_rows:  # written before the design rows, which mark the wave done
                ref_path = path.with_name(path.name.replace("raw_shard", "refruns_shard"))
                new_ref = not ref_path.exists()
                with ref_path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(ref_rows[0]))
                    if new_ref:
                        writer.writeheader()
                    writer.writerows(ref_rows)
            new_file = not path.exists()
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)
            print(f"budget {budget} reps {reps[0]}-{reps[-1]}: {time.perf_counter() - t0:.0f}s, "
                  f"cumulative calls {llm.calls}, output tokens {llm.output_tokens}", flush=True)


def pooled_reference() -> dict:
    """Each question's reference from the runs made inside its design waves (all shards)."""
    rows = {}
    for path in sorted(OUT.glob("refruns_shard*.csv")):
        for row in csv.DictReader(path.open()):
            rows[(row["item_id"], row["budget"], row["replication"])] = row  # a resumed wave may repeat a row
    counts, invalid, reps = {}, defaultdict(int), defaultdict(int)
    for (it, _, _), row in rows.items():
        counts[it] = counts.get(it, 0) + np.array([int(x) for x in row["final_counts"].split()])
        invalid[it] += int(row["invalid"])
        reps[it] += 1
    out = {}
    for q, item in enumerate(retained_items(every=True)):
        it = item["item_id"]
        if it not in counts:
            continue
        values = np.repeat(utilities(item)[1:], counts[it])
        rng = _rng(1, q, 1)
        boot = [_empirical_cvar(values[rng.integers(0, values.size, values.size)], ALPHA)
                for _ in range(int(CONFIG["reference"]["bootstrap_resamples"]))]
        wrong = sum(int(counts[it][s]) for s in range(STATES - 1) if output_coordinates(s)[0] != int(item["correct_index"]))
        out[it] = {"cvar": _empirical_cvar(values, ALPHA), "bootstrap_variance": float(np.var(boot, ddof=1)),
                   "runs": int(values.size), "replications": reps[it], "invalid": invalid[it],
                   "error_rate": wrong / values.size}
    (OUT / "reference_pooled.json").write_text(json.dumps({"items": out}, indent=1) + "\n")
    return out


def analyze() -> None:
    reference = {}
    for path in sorted(OUT.glob("reference_shard*.json")):
        reference.update(json.loads(path.read_text())["items"])
    if IN_WAVES:
        reference = pooled_reference()
    est = defaultdict(dict)
    calls = defaultdict(list)
    rows = [r for path in sorted(OUT.glob("raw_shard*.csv")) for r in csv.DictReader(path.open())]
    for row in rows:
        key = (row["item_id"], int(row["budget"]), row["design"])
        est[key][int(row["replication"])] = float(row["estimate"])
        calls[key].append(int(row["calls"]))
    items = sorted({row["item_id"] for row in rows})  # the questions actually run (the scale rule may use fewer)
    missing = [it for it in items if it not in reference]
    if missing:
        raise SystemExit(f"no reference for {missing}")
    budgets = CONFIG["budgets_calls_per_question"]
    designs = ["complete_rollout", "rollout_model", *DESIGNS]

    def sq(item, budget, design):
        e = est[(item, budget, design)]
        return np.array([(e[r] - reference[item]["cvar"]) ** 2 for r in sorted(e)])

    def corrected(item, budget, design):
        return sq(item, budget, design).mean() - reference[item]["bootstrap_variance"]

    summary = []
    for b in budgets:
        for d in designs:
            for it in items:
                s = sq(it, b, d)
                e = np.array(list(est[(it, b, d)].values()))
                summary.append({"item_id": it, "budget": b, "design": d, "replications": s.size,
                                "mse_uncorrected": float(s.mean()), "mse": float(corrected(it, b, d)),
                                "mse_se": float(s.std(ddof=1) / np.sqrt(s.size)),
                                "bias": float(e.mean() - reference[it]["cvar"]),
                                "mean_calls": float(np.mean(calls[(it, b, d)]))})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    rng = np.random.default_rng(SEED + 1)
    resamples = int(CONFIG["bootstrap_resamples"])

    def family(b, a, r, paired, subset=None):
        its = subset or items
        num = np.mean([corrected(it, b, a) for it in its])
        den = np.mean([corrected(it, b, r) for it in its])
        ba, br = [], []
        for it in its:
            sa, sr = sq(it, b, a), sq(it, b, r)
            ia = rng.integers(0, sa.size, size=(resamples, sa.size))
            ir = ia if paired else rng.integers(0, sr.size, size=(resamples, sr.size))
            v = reference[it]["bootstrap_variance"]
            ba.append(sa[ia].mean(axis=1) - v)
            br.append(sr[ir].mean(axis=1) - v)
        ba, br = np.mean(ba, axis=0), np.mean(br, axis=0)
        centred = (ba - br) - (num - den)
        unc = (np.mean([sq(it, b, a).mean() for it in its]) / np.mean([sq(it, b, r).mean() for it in its]))
        return {"mse_ratio": float(num / den), "ratio_ci95": [float(np.quantile(ba / br, .025)),
                                                              float(np.quantile(ba / br, .975))],
                "mse_difference": float(num - den), "uncorrected_ratio": float(unc),
                "bootstrap_p_two_sided": float(np.mean(np.abs(centred) >= abs(num - den)))}

    analysis = {"primary": {}, "secondary": {}}
    for name, p in CONFIG["primary"].items():
        analysis["primary"][name] = family(p["budget"], p["method"], p["reference"], p["paired"])
    ps = {k: max(v["bootstrap_p_two_sided"], 1.0 / resamples) for k, v in analysis["primary"].items()}
    adjusted, running = {}, 0.0
    for rank, key in enumerate(sorted(ps, key=ps.get)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[key]))
        adjusted[key] = running
    analysis["holm"] = {"raw": ps, "adjusted": adjusted}
    # The same tests over the first 12 retained questions.
    first = [it["item_id"] for it in retained_items(every=True)[:12]]
    if "scale" not in CONFIG and len(items) > 12 and set(first) <= set(items):
        analysis["first_12_questions"] = {name: family(p["budget"], p["method"], p["reference"], p["paired"], first)
                                          for name, p in CONFIG["primary"].items()}
    analysis["questions"] = {"count": len(items), "replications_per_question":
                             sorted({len(est[(it, b, d)]) for it in items for b in budgets for d in designs})}
    checkpoint = set(DESIGNS)
    for b in budgets:
        for a in designs:
            for r in designs:
                if a != r:
                    analysis["secondary"][f"{a}_vs_{r}|{b}"] = family(b, a, r, paired=(a in checkpoint and r in checkpoint))
    validation = []  # design-neutral: complete-rollout mean against the reference
    for it in items:
        e = np.array(list(est[(it, budgets[0], "complete_rollout")].values()))
        se = float(np.sqrt(e.var(ddof=1) / e.size + reference[it]["bootstrap_variance"]))
        validation.append({"item_id": it, "reference_cvar": reference[it]["cvar"], "rollout_mean": float(e.mean()),
                           "z": float((e.mean() - reference[it]["cvar"]) / se) if se > 1e-12 else 0.0,
                           "reference_error_rate": reference[it]["error_rate"]})
    (OUT / "validation.json").write_text(json.dumps(validation, indent=1) + "\n")
    passing = [v["item_id"] for v in validation if abs(v["z"]) <= 3]
    if IN_WAVES and len(passing) < len(items):  # the same tests without questions that fail the reference check
        analysis["primary_without_failing_questions"] = {
            "excluded": sorted(set(items) - set(passing)),
            **{name: family(p["budget"], p["method"], p["reference"], p["paired"], passing)
               for name, p in CONFIG["primary"].items()}}
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    print(json.dumps(analysis["primary"], indent=1), json.dumps(analysis["holm"], indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    step = args[0] if args else ""
    stub = "--stub" in args
    reps = int(args[args.index("--replications") + 1]) if "--replications" in args else None
    shard = tuple(int(x) for x in args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
    questions = tuple(int(x) for x in args[args.index("--questions") + 1].split(":")) if "--questions" in args else None
    if step == "smoke":
        smoke(stub)
    elif step == "decide":
        decide(int(args[1]))
    elif step == "screen" and "--shard" in args:
        positions = tuple(int(x) for x in args[args.index("--positions") + 1].split(":")) if "--positions" in args else None
        screen_shard(stub, shard, positions)
    elif step == "screen":
        screen(stub)
    elif step == "retain":
        retain()
    elif step == "reference":
        reference(stub, shard, questions)
    elif step == "run":
        run(stub, reps, shard, questions)
    elif step == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)
