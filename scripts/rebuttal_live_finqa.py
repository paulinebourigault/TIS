#!/usr/bin/env python3
"""FinQA review workflows with a typed state; `simulate` gives the fitted-model control of Appendix G.6.

Each call sees the report, the question, the eight candidate solutions and the current candidate with its
confidence, and returns a candidate letter and a confidence digit. In `run`, every query is a completion
sampled by the serving engine from the live model, and the evaluator sees samples only. The experiment logic
is that of src/tis/finqa/runner.py (pilot, floor, rounding, coupled streams, complete rollouts at matched
calls), written as waves so that many questions share one engine call.

  simulate   same logic, samples drawn from the stored exact laws (the control of Appendix G.6; CPU)
  power      how often an experiment of a given size detects the main comparisons (from `simulate` output)
  check      before any run: the engine's law and samples against the stored exact law on a few questions,
             and throughput (GPU)
  laws       the target: every prompt's exact law as computed on the GPU that serves the queries
  run        live runs (GPU); --shard k/K splits the replications
  analyze    errors against the exact CVaR under the target laws
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.finqa import CANDIDATE_LABELS, CONFIDENCE_LABELS, CONFIDENCES, OUTCOMES, STATES  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.finqa.protocol import render_messages  # noqa: E402
from tis.llm_study.common import read_json, read_jsonl  # noqa: E402
from tis.llm_study.runner import _anchored_score, _empirical_cvar, _mapping, _occupancy_scores  # noqa: E402
from tis.model import counts_from_streams, model_from_counts  # noqa: E402

CONFIG = json.loads((ROOT / os.environ.get("LIVE_FINQA_CONFIG", "configs/rebuttal_live_finqa_planning.json")).read_text())
OUT = Path(os.environ.get("LIVE_FINQA_OUT", ROOT / CONFIG["output_directory"]))
ALPHA = float(CONFIG["alpha"])
H = int(CONFIG["horizon"])
B = int(CONFIG["budget_per_group"])
SEED = int(CONFIG["master_seed"])
WORKFLOWS = list(CONFIG["workflows"])
STREAM_METHODS = list(CONFIG["stream_methods"])
ROLLOUT_METHODS = ["complete_rollout", "rollout_model"]
FIELDS = ["workflow", "panel_index", "replication", "method", "estimate", "calls"]


def stored_artifact(workflow: str) -> dict:
    return read_json(ROOT / CONFIG["kernels"][workflow])


def truth_artifact(workflow: str) -> dict:
    """The exact law computed on the run GPU by `laws` (the target); the stored law only in `simulate`."""
    path = OUT / f"laws_{workflow}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `laws` on the GPU that serves the queries")
    return read_json(path)


def question_models(artifact: dict) -> dict:
    banks = {int(b["panel_index"]): b["candidates"] for b in read_json(ROOT / CONFIG["banks"])["banks"]}
    return {int(q["panel_index"]): finqa_question_model(q, banks[int(q["panel_index"])], H)
            for q in artifact["questions"]}


def task(model):
    """One question and replication. Yields lists of (state, n); receives one array of outcomes per request.

    Only the model's structure is used (rewards and next state of each outcome, policy, grid): every
    probability comes from the samples received.
    """
    groups = tuple(model.query_groups)
    G = len(groups)
    total = G * B
    exploration = total ** (-float(CONFIG["exploration_exponent"]))
    minimum = int(CONFIG["minimum_main_per_group"])
    m = pilot_size(total, G, float(CONFIG["pilot_multiplier"]), int(CONFIG["minimum_pilot_per_group"]), minimum)
    main_total = total - G * m
    tally = np.zeros((G, OUTCOMES), dtype=np.int64)

    draws = yield [(s, m) for s in range(G)]
    pilot_counts = {g: np.bincount(d, minlength=OUTCOMES) for g, d in zip(groups, draws)}
    for s, g in enumerate(groups):
        tally[s] += pilot_counts[g]
    pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
    occupancy = _occupancy_scores(model, pilot)
    w_occ = regularized_weights(occupancy, exploration)
    w_mean = regularized_weights(pilot.mean_scales, exploration)
    weights = {
        "tis": regularized_weights(pilot.tail_scales, exploration),
        "tis_anchored": regularized_weights(_anchored_score(pilot.tail_scales, occupancy), exploration),
        "learned_occupancy": w_occ,
        "learned_mean": w_mean,
        "occ_plus_mean": 0.5 * w_mean + 0.5 * w_occ,
        "occ_plus_uniform": 0.5 * np.full(G, 1.0 / G) + 0.5 * w_occ,
    }
    counts_by = {}
    for method in STREAM_METHODS:
        if method == "uniform":
            counts_by[method] = integer_allocation(total, np.full(G, 1.0 / G), minimum)
        else:
            counts_by[method] = integer_allocation(main_total, weights[method], minimum)
    maximum = np.max(np.vstack([counts_by[k] for k in STREAM_METHODS]), axis=0)

    draws = yield [(s, int(maximum[s])) for s in range(G)]
    streams = dict(zip(groups, draws))
    for s, d in enumerate(draws):
        tally[s] += np.bincount(d, minlength=OUTCOMES)
    rows = []
    for method in STREAM_METHODS:
        empirical = counts_from_streams(model, streams, _mapping(model, counts_by[method]))
        estimate = solve_stop_loss(model_from_counts(model, empirical), ALPHA).cvar
        charged = int(counts_by[method].sum()) + (0 if method == "uniform" else G * m)
        rows.append((method, float(estimate), charged))

    runs = total // H
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    rollout_counts = np.zeros((G, OUTCOMES), dtype=np.int64)
    for _ in range(H):
        current = [int(s) for s in np.unique(state)]
        draws = yield [(s, int((state == s).sum())) for s in current]
        successor = np.empty_like(state)
        for s, d in zip(current, draws):
            mask = state == s
            kernel = model.kernels[groups[s]]
            totals[mask] += kernel.rewards[d]
            successor[mask] = kernel.next_states[d]
            rollout_counts[s] += np.bincount(d, minlength=OUTCOMES)
        state = successor
    tally += rollout_counts
    rows.append(("complete_rollout", float(_empirical_cvar(totals, ALPHA)), runs * H))
    # the same runs under TIS's estimator; a state no run visited is unreachable in the fitted model
    filled = {g: (rollout_counts[s] if rollout_counts[s].sum() > 0 else np.ones(OUTCOMES, dtype=np.int64))
              for s, g in enumerate(groups)}
    rows.append(("rollout_model", float(solve_stop_loss(model_from_counts(model, filled), ALPHA).cvar), runs * H))
    return rows, tally


def drive(tasks: dict, sampler) -> dict:
    """Advance all tasks wave by wave; one sampler call per wave. tasks: key (workflow, q) -> generator."""
    results, requests = {}, {}
    for key, generator in tasks.items():
        requests[key] = next(generator)
    while requests:
        flat = [(key[0], key[1], s, n) for key, reqs in requests.items() for s, n in reqs]
        draws = sampler.draw(flat)
        position, following = 0, {}
        for key, reqs in requests.items():
            got = draws[position:position + len(reqs)]
            position += len(reqs)
            try:
                following[key] = tasks[key].send(got)
            except StopIteration as stop:
                results[key] = stop.value
        requests = following
    return results


class LawSampler:
    """Samples from given laws (the stored exact laws in `simulate`)."""

    def __init__(self, laws: dict, rng: np.random.Generator):
        self.laws, self.rng, self.queries = laws, rng, 0

    def draw(self, requests):
        self.queries += sum(n for *_, n in requests)
        return [self.rng.choice(OUTCOMES, size=n, p=self.laws[(w, q)][s]) for w, q, s, n in requests]


def stored_laws(workflows) -> dict:
    laws = {}
    for workflow in workflows:
        for q in stored_artifact(workflow)["questions"]:
            laws[(workflow, int(q["panel_index"]))] = {
                int(k["state_index"]): np.asarray(k["probabilities"], dtype=float) for k in q["kernels"]}
    return laws


class EngineSampler:
    """Live sampling through vLLM: the candidate letter, then the confidence digit given that letter.

    Each is one sampled token restricted to its labels at the protocol temperature, which is the constrained
    two-token decoder of the paper's protocol. The engine returns tokens only.
    """

    def __init__(self, seed: int):
        from vllm import LLM, SamplingParams

        spec = read_json(ROOT / CONFIG["protocol"])
        m = CONFIG["model"]
        assert m["identifier"] == spec["generator"]["identifier"] and m["revision"] == spec["generator"]["revision"]
        self.temperature = float(spec["decoder"]["temperature"])
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=m["identifier"], revision=m["revision"], tokenizer_revision=m["revision"],
                       dtype=m["dtype"], seed=seed, gpu_memory_utilization=float(m["gpu_memory_utilization"]),
                       max_model_len=int(m["max_model_len"]), enable_prefix_caching=True,
                       max_num_seqs=int(m["max_num_seqs"]), max_logprobs=int(m["top_logprobs"]),
                       **m.get("engine_kwargs", {}))
        self.top_logprobs, self.logprob_modes = int(m["top_logprobs"]), set()
        self.tokenizer = self.llm.get_tokenizer()
        self.candidate_ids = self._single(CANDIDATE_LABELS)
        self.confidence_ids = self._single(CONFIDENCE_LABELS)
        self.panel = {int(item["panel_index"]): item for item in read_jsonl(ROOT / CONFIG["panel"])}
        self.banks = {int(b["panel_index"]): b["candidates"] for b in read_json(ROOT / CONFIG["banks"])["banks"]}
        self.prompts, self.queries, self.seconds = {}, 0, 0.0

    def _single(self, labels):
        ids = []
        for label in labels:
            encoded = self.tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"label {label!r} is not a single token")
            ids.append(int(encoded[0]))
        assert len(set(ids)) == len(ids)
        return ids

    def prompt(self, workflow: str, q: int, state: int) -> list:
        key = (workflow, q, state)
        if key not in self.prompts:
            messages = render_messages(self.panel[q], self.banks[q], state, workflow)
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            self.prompts[key] = [int(t) for t in ids]
        return self.prompts[key]

    def _tokens(self, prompts, counts, allowed):
        index = {token: i for i, token in enumerate(allowed)}
        params = [self.SamplingParams(n=int(c), max_tokens=1, temperature=self.temperature, top_p=1.0,
                                      allowed_token_ids=allowed, detokenize=False) for c in counts]
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        result = []
        for output, c in zip(outputs, counts):
            tokens = [index[int(o.token_ids[0])] for o in output.outputs]
            assert len(tokens) == c
            result.append(np.asarray(tokens, dtype=int))
        return result

    def _label_laws(self, prompts, allowed):
        """Each prompt's next-token law restricted to `allowed`, from the engine's log-probabilities.

        The engine reports either raw log-probabilities (before the restriction and the temperature; other
        tokens then have finite values) or processed ones (after both). Returns None where a label is missing.
        """
        params = self.SamplingParams(n=1, max_tokens=1, temperature=self.temperature, top_p=1.0,
                                     allowed_token_ids=allowed, logprobs=self.top_logprobs, detokenize=False)
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        laws = []
        for output in outputs:
            reported = output.outputs[0].logprobs[0]
            values = [reported[t].logprob if t in reported else None for t in allowed]
            raw = any(np.isfinite(v.logprob) for t, v in reported.items() if t not in allowed)
            self.logprob_modes.add("raw" if raw else "processed")
            if any(v is None or not np.isfinite(v) for v in values):
                laws.append(None)
                continue
            scaled = np.asarray(values, dtype=float) / (self.temperature if raw else 1.0)
            weights = np.exp(scaled - scaled.max())
            laws.append(weights / weights.sum())
        return laws

    def laws(self, keys):
        """Exact 72-outcome law of each (workflow, question, state) as this engine computes it, or None."""
        prompts = [self.prompt(*key) for key in keys]
        letters = self._label_laws(prompts, self.candidate_ids)
        second = [p + [c] for p in prompts for c in self.candidate_ids]
        digits = self._label_laws(second, self.confidence_ids)
        result = []
        for i, first in enumerate(letters):
            rows = digits[i * len(self.candidate_ids):(i + 1) * len(self.candidate_ids)]
            if first is None or any(r is None for r in rows):
                result.append(None)
                continue
            joint = (first[:, None] * np.vstack(rows)).reshape(-1)
            result.append(joint / joint.sum())
        return result

    def draw(self, requests):
        started = time.perf_counter()
        prompts = [self.prompt(w, q, s) for w, q, s, _ in requests]
        letters = self._tokens(prompts, [n for *_, n in requests], self.candidate_ids)
        second_prompts, second_counts, where = [], [], []
        for i, drawn in enumerate(letters):
            for c in np.unique(drawn):
                second_prompts.append(prompts[i] + [self.candidate_ids[int(c)]])
                second_counts.append(int((drawn == c).sum()))
                where.append((i, int(c)))
        digits = self._tokens(second_prompts, second_counts, self.confidence_ids)
        result = [np.empty(len(drawn), dtype=int) for drawn in letters]
        for (i, c), d in zip(where, digits):
            # digits fill the positions of their letter in the engine's order, so every prefix is exchangeable
            result[i][letters[i] == c] = c * CONFIDENCES + d
        self.queries += sum(n for *_, n in requests)
        self.seconds += time.perf_counter() - started
        return result


def run_replication(replication: int, sampler, models: dict):
    tasks = {(w, q): task(model) for w in WORKFLOWS for q, model in models[w].items()}
    results = drive(tasks, sampler)
    rows, tallies = [], {}
    for (w, q), (task_rows, tally) in results.items():
        tallies[(w, q)] = tally
        for method, estimate, calls in task_rows:
            rows.append({"workflow": w, "panel_index": q, "replication": replication, "method": method,
                         "estimate": repr(estimate), "calls": calls})
    return rows, tallies


def _simulate_job(replication: int):
    models = {w: question_models(stored_artifact(w)) for w in WORKFLOWS}
    sampler = LawSampler(stored_laws(WORKFLOWS), np.random.default_rng([SEED, 7, replication]))
    rows, _ = run_replication(replication, sampler, models)
    return rows


def simulate(replications: int, processes: int) -> None:
    from multiprocessing import Pool

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with Pool(processes) as pool, (OUT / "raw_simulated.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for done, rows in enumerate(pool.imap_unordered(_simulate_job, range(replications)), 1):
            writer.writerows(rows)
            stream.flush()
            if done % 10 == 0:
                print(f"{done} replications, {time.time() - started:.0f}s", flush=True)


def run(shard: int, shards: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"raw_shard{shard}of{shards}.csv"
    tally_path = OUT / f"tallies_shard{shard}of{shards}.npz"
    done = set()
    if raw.exists():
        done = {int(r["replication"]) for r in csv.DictReader(raw.open())}
    tallies = {}
    if tally_path.exists():
        with np.load(tally_path) as stored:
            tallies = {k: stored[k] for k in stored.files}
    models = {w: question_models(stored_artifact(w)) for w in WORKFLOWS}
    sampler = EngineSampler(SEED + 1000 * shards + shard)
    todo = [r for r in range(int(CONFIG["replications"])) if r % shards == shard and r not in done]
    for replication in todo:
        started = time.perf_counter()
        rows, new = run_replication(replication, sampler, models)
        for (w, q), tally in new.items():
            key = f"{w}|{q}"
            tallies[key] = tallies.get(key, 0) + tally
        temporary = tally_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **tallies)
        fresh = not raw.exists()
        with raw.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if fresh:
                writer.writeheader()
            writer.writerows(rows)
        temporary.replace(tally_path)  # a crash between the two writes repeats the replication's tally at most once
        print(f"replication {replication}: {time.perf_counter() - started:.0f}s, engine {sampler.seconds:.0f}s, "
              f"cumulative queries {sampler.queries}", flush=True)


def _binomial_p(count: int, n: int, p: float) -> float:
    from tis.finqa.calibrate import _binomial_two_sided_p  # the exact test of the paper's generation audit

    return float(_binomial_two_sided_p(int(count), int(n), float(min(max(p, 0.0), 1.0))))


def compare_with_law(tallies: dict, laws: dict) -> dict:
    """Every outcome's sampled frequency against the law: standardized gaps, and the exact binomial test
    (Bonferroni over all cells, family level .01) for every cell whose gap exceeds three standard errors."""
    cells, worst, largest_z, minimum_p, exact = 0, 0.0, 0.0, 1.0, []
    for (w, q), tally in tallies.items():
        for s in range(STATES):
            n = int(tally[s].sum())
            if n == 0:
                continue
            law = np.asarray(laws[(w, q)][s], dtype=float)
            cells += OUTCOMES
            worst = max(worst, float(np.abs(tally[s] / n - law).max()))
            z = (tally[s] - n * law) / np.sqrt(np.maximum(n * law * (1 - law), 1e-300))
            largest_z = max(largest_z, float(np.abs(z).max()))
            for j in np.flatnonzero(np.abs(z) > 3):
                p = _binomial_p(tally[s][j], n, law[j])
                minimum_p = min(minimum_p, p)
                exact.append([w, q, s, int(j), int(tally[s][j]), n, float(law[j]), p])
    threshold = 0.01 / max(cells, 1)
    return {"cells": cells, "largest_frequency_gap": worst, "largest_standardized_gap": largest_z,
            "cells_tested_exactly": len(exact), "smallest_exact_p": minimum_p,
            "bonferroni_threshold_family_.01": threshold,
            "cells_below_threshold": [e for e in exact if e[-1] < threshold]}


def law_cvar(model, law: dict) -> float:
    replaced = model.replace_query_probabilities({g: law[s] for s, g in enumerate(model.query_groups)})
    return float(solve_stop_loss(replaced, ALPHA).cvar)


def _law_table(artifact: dict) -> dict:
    return {int(q["panel_index"]): {int(k["state_index"]): np.asarray(k["probabilities"], dtype=float)
                                    for k in q["kernels"]} for q in artifact["questions"]}


def _law_gaps(models: dict, new: dict, old: dict) -> dict:
    """Largest probability gap and largest CVaR gap between two sets of laws (same questions)."""
    gap = max(float(np.abs(new[q][s] - old[q][s]).max()) for q in new for s in range(STATES))
    cvar = max(abs(law_cvar(models[q], new[q]) - law_cvar(models[q], old[q])) for q in new)
    return {"largest_probability_gap": gap, "largest_cvar_gap": cvar}


def transformers_laws(selected, name: str) -> dict:
    """The paper's calibration code (transformers, same weights) on this GPU, for the selected questions."""
    import gc

    import torch

    from tis.finqa.calibrate import calibrate_kernels

    panel = [item for item in read_jsonl(ROOT / CONFIG["panel"]) if int(item["panel_index"]) in set(selected)]
    subset = OUT / f"{name}_panel.jsonl"
    subset.write_text("".join(json.dumps(item) + "\n" for item in panel))
    out = {}
    for w in WORKFLOWS:
        out[w] = calibrate_kernels(ROOT / CONFIG["protocol"], subset, ROOT / CONFIG["banks"],
                                   OUT / f"{name}_{w}.json", w)
    gc.collect()
    torch.cuda.empty_cache()
    return out


def engine_laws(sampler, selected) -> dict:
    """Engine laws per workflow as artifacts in the stored format; a workflow maps to None if any law is missing."""
    out = {}
    for w in WORKFLOWS:
        stored = stored_artifact(w)
        questions, complete = [], True
        for q in stored["questions"]:
            index = int(q["panel_index"])
            if index not in set(selected):
                continue
            laws = sampler.laws([(w, index, s) for s in range(STATES)])
            if any(law is None for law in laws):
                complete = False
                break
            questions.append({"panel_index": index, "item_id": q["item_id"], "gold_answer": q["gold_answer"],
                              "kernels": [{"state_index": s, "action": k["action"], "probabilities": laws[s].tolist()}
                                          for s, k in enumerate(q["kernels"])]})
        out[w] = {"workflow": w, "source": "engine log-probabilities", "logprob_modes": sorted(sampler.logprob_modes),
                  "questions": questions} if complete else None
    return out


def check(questions: int, samples: int, recompute: bool) -> None:
    """Before any run: do the engine's law and samples agree with the stored exact law, and how fast is it?"""
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"questions": questions, "samples_per_prompt": samples}
    models = {w: question_models(stored_artifact(w)) for w in WORKFLOWS}
    stored = {w: _law_table(stored_artifact(w)) for w in WORKFLOWS}
    selected = sorted(models[WORKFLOWS[0]])[:questions]
    if recompute:
        recomputed = transformers_laws(selected, "check_transformers")
        report["transformers_on_this_gpu_against_stored"] = {
            w: _law_gaps(models[w], _law_table(recomputed[w]), stored[w]) for w in WORKFLOWS}
        print(json.dumps(report["transformers_on_this_gpu_against_stored"], indent=1), flush=True)
    sampler = EngineSampler(SEED + 99)
    from_engine = engine_laws(sampler, selected)
    report["engine_logprob_modes"] = sorted(sampler.logprob_modes)
    report["engine_law_against_stored"] = {
        w: (_law_gaps(models[w], _law_table(from_engine[w]), stored[w]) if from_engine[w] else "label missing from log-probabilities")
        for w in WORKFLOWS}
    print(json.dumps({k: report[k] for k in ("engine_logprob_modes", "engine_law_against_stored")}, indent=1), flush=True)
    requests = [(w, q, s, samples) for w in WORKFLOWS for q in selected for s in range(STATES)]
    started = time.perf_counter()
    draws = sampler.draw(requests)
    elapsed = time.perf_counter() - started
    tallies = defaultdict(lambda: np.zeros((STATES, OUTCOMES), dtype=np.int64))
    for (w, q, s, _), d in zip(requests, draws):
        tallies[(w, q)][s] += np.bincount(d, minlength=OUTCOMES)
    report["samples_against_stored_law"] = compare_with_law(
        tallies, {(w, q): stored[w][q] for w in WORKFLOWS for q in selected})
    if all(from_engine.values()):
        report["samples_against_engine_law"] = compare_with_law(
            tallies, {(w, q): _law_table(from_engine[w])[q] for w in WORKFLOWS for q in selected})
    report["throughput"] = {"queries": len(requests) * samples, "seconds": elapsed,
                            "queries_per_second": len(requests) * samples / elapsed}
    report["cvar"] = {f"{w}|{q}": {"stored_law": law_cvar(models[w][q], stored[w][q]),
                                   "engine_samples": law_cvar(models[w][q], {s: tally[s] / tally[s].sum() for s in range(STATES)})}
                      for (w, q), tally in tallies.items()}
    started, before = time.perf_counter(), sampler.queries
    results = drive({(w, q): task(models[w][q]) for w in WORKFLOWS for q in selected}, sampler)
    report["one_replication"] = {
        "seconds": time.perf_counter() - started, "queries": sampler.queries - before,
        "estimates": {f"{w}|{q}": {m: e for m, e, _ in rows} for (w, q), (rows, _) in results.items()}}
    (OUT / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cvar", "one_replication")}, indent=1))
    print("one replication of the selected questions:", round(report["one_replication"]["seconds"]), "s,",
          report["one_replication"]["queries"], "queries")


def laws() -> None:
    """The target laws on the GPU that serves the queries: the engine's own law for every prompt
    (from its log-probabilities); if a label is ever missing there, the law from the paper's calibration code."""
    OUT.mkdir(parents=True, exist_ok=True)
    every = sorted(question_models(stored_artifact(WORKFLOWS[0])))
    sampler = EngineSampler(SEED + 98)
    from_engine = engine_laws(sampler, every)
    report = {"logprob_modes": sorted(sampler.logprob_modes)}
    if all(from_engine.values()):
        for w in WORKFLOWS:
            (OUT / f"laws_{w}.json").write_text(json.dumps(from_engine[w]) + "\n")
        report["target"] = "engine log-probabilities"
    else:
        del sampler
        recomputed = transformers_laws(every, "laws_transformers")
        for w in WORKFLOWS:
            recomputed[w]["source"] = "transformers calibration code on the run GPU"
            (OUT / f"laws_{w}.json").write_text(json.dumps(recomputed[w]) + "\n")
        report["target"] = "transformers calibration code (a label was missing from the engine's log-probabilities)"
    for w in WORKFLOWS:
        models = question_models(stored_artifact(w))
        report[w] = _law_gaps(models, _law_table(read_json(OUT / f"laws_{w}.json")), _law_table(stored_artifact(w)))
    (OUT / "laws_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def load_rows(simulated: bool):
    paths = [OUT / "raw_simulated.csv"] if simulated else sorted(OUT.glob("raw_shard*.csv"))
    est = defaultdict(dict)
    for path in paths:
        for r in csv.DictReader(path.open()):
            est[(r["workflow"], int(r["panel_index"]), r["method"])][int(r["replication"])] = float(r["estimate"])
    return est


def family(sq: dict, questions, a: str, b: str, paired: bool, rng, resamples: int) -> dict:
    """Ratio of equal-question-weight MSEs; bootstrap over replications within question; centred test."""
    num = np.mean([sq[(q, a)].mean() for q in questions])
    den = np.mean([sq[(q, b)].mean() for q in questions])
    ba, bb = 0.0, 0.0
    for q in questions:
        sa, sb = sq[(q, a)], sq[(q, b)]
        ia = rng.integers(0, sa.size, size=(resamples, sa.size))
        ib = ia if paired else rng.integers(0, sb.size, size=(resamples, sb.size))
        ba = ba + sa[ia].mean(axis=1) / len(questions)
        bb = bb + sb[ib].mean(axis=1) / len(questions)
    centred = (ba - bb) - (num - den)
    return {"mse_ratio": float(num / den), "ratio_ci95": [float(np.quantile(ba / bb, .025)), float(np.quantile(ba / bb, .975))],
            "mse_a": float(num), "mse_b": float(den),
            "bootstrap_p_two_sided": float(np.mean(np.abs(centred) >= abs(num - den)))}


def squared_errors(est: dict, workflow: str, truth: dict, replications=None) -> tuple:
    questions = sorted({q for w, q, _ in est if w == workflow})
    sq = {}
    for (w, q, method), values in est.items():
        if w != workflow:
            continue
        reps = sorted(values) if replications is None else replications
        sq[(q, method)] = np.array([(values[r] - truth[(workflow, q)]) ** 2 for r in reps])
    return questions, sq


def exact_truth(simulated: bool) -> dict:
    truth = {}
    for w in WORKFLOWS:
        artifact = stored_artifact(w) if simulated else truth_artifact(w)
        for q, model in question_models(artifact).items():
            truth[(w, q)] = float(solve_stop_loss(model, ALPHA).cvar)
    return truth


def analyze(simulated: bool) -> None:
    est = load_rows(simulated)
    truth = exact_truth(simulated)
    rng = np.random.default_rng(SEED + 1)
    resamples = int(CONFIG["bootstrap_resamples"])
    analysis = {"primary": {}, "secondary": {}, "replications": {}}
    stream = set(STREAM_METHODS)
    for w in WORKFLOWS:
        questions, sq = squared_errors(est, w, truth)
        analysis["replications"][w] = sorted({v.size for v in sq.values()})
        methods = sorted({m for _, m in sq})
        for a in methods:
            for b in methods:
                if a != b and (a in stream) and b in ("complete_rollout", "rollout_model", "occ_plus_mean", "tis_anchored", "uniform", "learned_occupancy"):
                    paired = (a in stream) == (b in stream)
                    analysis["secondary"][f"{w}|{a}_vs_{b}"] = family(sq, questions, a, b, paired, rng, resamples)
        analysis["secondary"][f"{w}|rollout_model_vs_complete_rollout"] = family(
            sq, questions, "rollout_model", "complete_rollout", True, rng, resamples)
    for name, p in CONFIG["primary"].items():
        questions, sq = squared_errors(est, p["workflow"], truth)
        analysis["primary"][name] = family(sq, questions, p["method"], p["reference"], False, rng, resamples)
    ps = {k: max(v["bootstrap_p_two_sided"], 1.0 / resamples) for k, v in analysis["primary"].items()}
    adjusted, running = {}, 0.0
    for rank, key in enumerate(sorted(ps, key=ps.get)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[key]))
        adjusted[key] = running
    analysis["holm"] = {"raw": ps, "adjusted": adjusted}
    # validity: the mean complete-rollout estimate against the exact CVaR, question by question
    validation = []
    for w in WORKFLOWS:
        for q in sorted({q for ww, q, _ in est if ww == w}):
            values = np.array(list(est[(w, q, "complete_rollout")].values()))
            spread = values.std(ddof=1) / np.sqrt(values.size)
            z = 0.0 if spread == 0 and abs(values.mean() - truth[(w, q)]) < 1e-12 else float((values.mean() - truth[(w, q)]) / max(spread, 1e-300))
            validation.append({"workflow": w, "panel_index": q, "exact_cvar": truth[(w, q)],
                               "rollout_mean": float(values.mean()), "z": z})
    analysis["validation"] = {"largest_abs_z": max(abs(v["z"]) for v in validation),
                              "questions_with_abs_z_above_3": [(v["workflow"], v["panel_index"]) for v in validation if abs(v["z"]) > 3]}
    if not simulated:
        tallies = {}
        for path in sorted(OUT.glob("tallies_shard*.npz")):
            with np.load(path) as stored:
                for key in stored.files:
                    w, q = key.split("|")
                    tallies[(w, int(q))] = tallies.get((w, int(q)), 0) + stored[key]
        if tallies:
            laws = {}
            for w in WORKFLOWS:
                for q in truth_artifact(w)["questions"]:
                    laws[(w, int(q["panel_index"]))] = {int(k["state_index"]): np.asarray(k["probabilities"]) for k in q["kernels"]}
            analysis["engine_against_law"] = compare_with_law(tallies, laws)
            analysis["engine_samples"] = int(sum(int(t.sum()) for t in tallies.values()))
    name = "analysis_simulated.json" if simulated else "analysis.json"
    (OUT / name).write_text(json.dumps(analysis, indent=1) + "\n")
    (OUT / ("validation_simulated.json" if simulated else "validation.json")).write_text(json.dumps(validation, indent=1) + "\n")
    print(json.dumps({"primary": analysis["primary"], "holm": analysis["holm"], "validation": analysis["validation"],
                      "replications": analysis["replications"]}, indent=1))
    for key, v in analysis["secondary"].items():
        print(f"  {key}: {v['mse_ratio']:.3f} [{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}]")


def power(size: int, experiments: int) -> None:
    """Experiments of `size` replications drawn from the simulated ones: the main comparisons' ratios and CIs."""
    est = load_rows(True)
    truth = exact_truth(True)
    rng = np.random.default_rng(SEED + 2)
    available = sorted(next(iter(est.values())))
    out = {}
    for name, p in CONFIG["primary"].items():
        ratios, uppers = [], []
        for _ in range(experiments):
            reps = list(rng.choice(available, size=size, replace=False))
            questions, sq = squared_errors(est, p["workflow"], truth, reps)
            result = family(sq, questions, p["method"], p["reference"], False, rng, 2000)
            ratios.append(result["mse_ratio"])
            uppers.append(result["ratio_ci95"][1])
        out[name] = {"replications": size, "ratio_median": float(np.median(ratios)),
                     "ratio_5_95": [float(np.quantile(ratios, .05)), float(np.quantile(ratios, .95))],
                     "ci_upper_median": float(np.median(uppers)), "ci_upper_max": float(np.max(uppers)),
                     "share_with_ci_below_1": float(np.mean(np.array(uppers) < 1)),
                     "share_with_ci_below_.8": float(np.mean(np.array(uppers) < .8))}
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--replications", type=int, default=int(CONFIG["replications"]))
    s.add_argument("--processes", type=int, default=8)
    p = sub.add_parser("power")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--experiments", type=int, default=200)
    c = sub.add_parser("check")
    c.add_argument("--questions", type=int, default=3)
    c.add_argument("--samples", type=int, default=20000)
    c.add_argument("--no-recompute", action="store_true")
    sub.add_parser("laws")
    r = sub.add_parser("run")
    r.add_argument("--shard", default="0/1")
    a = sub.add_parser("analyze")
    a.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    if args.command == "simulate":
        simulate(args.replications, args.processes)
    elif args.command == "power":
        power(args.size, args.experiments)
    elif args.command == "check":
        check(args.questions, args.samples, not args.no_recompute)
    elif args.command == "laws":
        laws()
    elif args.command == "run":
        k, n = (int(x) for x in args.shard.split("/"))
        run(k, n)
    else:
        analyze(args.simulated)


if __name__ == "__main__":
    main()
