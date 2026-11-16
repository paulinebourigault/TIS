#!/usr/bin/env python3
"""A language model as the ordering policy of the paper's inventory simulator (Appendix I).

One query at (periods remaining, stock): the model, shown the current state only, samples an order (one digit,
restricted to the feasible orders, at the protocol temperature); the simulator then draws demand and disruption
and returns the period's profit and the next stock. The evaluator can set the simulator to any state. Dynamics,
profit and encoding are those of scripts/rebuttal_inventory_profit.py.

  prompts    print the prompts
  synthetic  write a synthetic decision law (tests; no model)
  policy     the model's exact decision probabilities in every state, from the serving engine's
             log-probabilities and, as a cross-check, through transformers (GPU)
  predict    first-order MSE of each design relative to fitted-model rollouts (instant)
  simulate   the experiment with orders drawn from the decision probabilities (CPU)
  check      engine-sampled orders against the decision probabilities; throughput (GPU)
  run        live: every order is sampled by the serving engine (GPU); --shard k/K splits the replications
  analyze    errors against the exact CVaR implied by the decision probabilities
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
sys.path.insert(0, str(ROOT / "scripts"))

import rebuttal_inventory_profit as B  # noqa: E402  (profit function, bounds, grid of the inventory family)
from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.environments import (  # noqa: E402
    INVENTORY_CAPACITY as CAP, INVENTORY_POISSON_MEAN, _censored_poisson_probabilities,
)
from tis.llm_study.runner import _empirical_cvar, _mapping  # noqa: E402
from tis.model import FiniteHorizonModel, Kernel, counts_from_streams, model_from_counts  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402

CONFIG = json.loads((ROOT / os.environ.get("LLM_INVENTORY_CONFIG", "configs/rebuttal_llm_inventory.json")).read_text())
OUT = Path(os.environ.get("LLM_INVENTORY_OUT", ROOT / CONFIG["output_directory"]))
H = B.HORIZON
ALPHAS = [float(a) for a in CONFIG["alphas"]]
BUDGET = int(CONFIG["budget_per_group"])
SEED = int(CONFIG["master_seed"])
PILOT = CONFIG["pilot"]
STREAM_METHODS = list(CONFIG["stream_methods"])
ENVIRONMENTS = [(float(p), int(loss)) for p in B.FAMILY["base_disruption_probabilities"]
                for loss in B.FAMILY["disruption_losses"]]
STATES = [(h, s) for h in range(1, H + 1) for s in range(CAP + 1)]
FIELDS = ["model", "environment", "replication", "alpha", "method", "error_profit"]

SYSTEM_PROMPT = (
    "You manage the inventory of one product over 8 periods. In each period you decide how many units to "
    "order. Reply with a single digit: the number of units to order now. No other text."
)
USER_TEMPLATE = (
    "Period {period} of 8.\n"
    "Units in stock: {stock}. Storage capacity: 6 units.\n"
    "Expected demand this period: about {demand} units. Demand varies randomly from period to period.\n"
    "Each unit sold earns 4. Each unit ordered costs 1.5. Each unit left in stock at the end of the period "
    "costs 0.5. Each unit of demand that cannot be served costs 2.\n"
    "Orders arrive immediately, before demand. Occasionally a disruption destroys part of the stock left at "
    "the end of a period.\n"
    "How many units do you order now? Answer with one digit from 0 to {largest}."
)


def messages(h: int, stock: int) -> list:
    user = USER_TEMPLATE.format(period=H - h + 1, stock=stock, demand=f"{INVENTORY_POISSON_MEAN[H - h]:g}",
                                largest=CAP - stock)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ----------------------------------------------------------------------------- environment
def structure(p_base: float, loss: int):
    """An environment's model with every feasible order possible, and per state the outcome index and the
    simulator probability of each (order, demand, disruption). The model's probabilities are placeholders:
    every use replaces them by a decision law's exact kernels or by sample counts."""
    kernels, tables = {}, {}
    for h in range(1, H + 1):
        demand_probabilities = _censored_poisson_probabilities(INVENTORY_POISSON_MEAN[H - h])
        for stock in range(CAP + 1):
            entries = []
            for order in range(CAP - stock + 1):
                available = stock + order
                disruption = p_base + 0.02 if available == CAP else p_base
                for demand, p_demand in enumerate(demand_probabilities):
                    sales, unmet, ending = min(available, demand), max(demand - available, 0), max(available - demand, 0)
                    profit = B.period_profit(sales, order, ending, unmet)
                    if not B.LOW <= profit <= B.HIGH:
                        raise ValueError("profit outside the declared bounds")
                    reward = (profit - B.LOW) / B.SPAN
                    for disrupted, p_event in ((0, 1.0 - disruption), (1, disruption)):
                        key = (float(reward), int(max(ending - loss * disrupted, 0)))
                        entries.append((order, demand, disrupted, key, float(p_demand * p_event)))
            ordered = sorted({e[3] for e in entries})
            index = {key: i for i, key in enumerate(ordered)}
            table = np.zeros((CAP - stock + 1, len(demand_probabilities), 2), dtype=int)
            simulator = np.zeros(table.shape)
            for order, demand, disrupted, key, p in entries:
                table[order, demand, disrupted] = index[key]
                simulator[order, demand, disrupted] = p
            kernels[(h, stock)] = Kernel(rewards=np.asarray([k[0] for k in ordered]),
                                         next_states=np.asarray([k[1] for k in ordered], dtype=int),
                                         probabilities=np.full(len(ordered), 1.0 / len(ordered)))
            tables[(h, stock)] = (table, simulator)
    reachable = {H: {0}}
    for h in range(H, 1, -1):
        reachable[h - 1] = {int(s) for state in reachable[h] for s in kernels[(h, state)].next_states}
    groups = tuple((h, s) for h in range(1, H + 1) for s in sorted(reachable[h]))
    policy = np.zeros((H + 1, CAP + 1, 1))
    policy[1:, :, 0] = 1.0
    model = FiniteHorizonModel(
        name=f"llm_inventory_p{p_base:g}_L{loss}", horizon=H, initial_state=0,
        grid=np.arange(H * B.DENOM + 1) / B.DENOM, policy=policy, kernels=kernels, query_groups=groups,
        group_mode="untied_layer_state", metadata={"p_base": p_base, "loss": loss})
    return model, tables


def exact_kernels(model, tables, law: dict) -> dict:
    """Exact outcome probabilities of every queried state under a decision law."""
    out = {}
    for group in model.query_groups:
        table, simulator = tables[group]
        weights = np.asarray(law[group], dtype=float)[:, None, None] * simulator
        out[group] = np.bincount(table.ravel(), weights=weights.ravel(), minlength=model.kernels[group].size)
    return out


def read_law(name: str) -> dict:
    stored = json.loads((OUT / f"policy_{name}.json").read_text())
    return {(int(k.split("|")[0]), int(k.split("|")[1])): np.asarray(v, dtype=float) for k, v in stored["laws"].items()}


# ----------------------------------------------------------------------------- one environment and replication
def task(model):
    """Yields lists of (group, n); receives one array of outcome indices per request. Uses the model's
    structure only (reward and next stock of each outcome, grid): every probability comes from the samples."""
    groups = tuple(model.query_groups)
    G = len(groups)
    total = G * BUDGET
    exploration = total ** (-float(PILOT["exploration_exponent"]))
    minimum = int(PILOT["minimum_main_per_group"])
    m = pilot_size(total, G, float(PILOT["multiplier"]), int(PILOT["minimum_pilot_per_group"]), minimum)
    main_total = total - G * m

    draws = yield [(g, m) for g in groups]
    pilot_counts = {g: np.bincount(d, minlength=model.kernels[g].size) for g, d in zip(groups, draws)}
    fitted = model_from_counts(model, pilot_counts)
    allocations = {}
    for alpha in ALPHAS:
        pilot = compute_influences(fitted, alpha)
        occupancy = _occupancy_scores(model, pilot.occupancy)
        tail, mean = np.asarray(pilot.tail_scales), np.asarray(pilot.mean_scales)
        weights = {
            "tis": regularized_weights(tail, exploration),
            "tis_anchored": regularized_weights(_anchored_score(tail, occupancy), exploration),
            "learned_occupancy": regularized_weights(occupancy, exploration),
            "learned_mean": regularized_weights(mean, exploration),
            "occ_plus_mean": 0.5 * regularized_weights(mean, exploration) + 0.5 * regularized_weights(occupancy, exploration),
        }
        for method in STREAM_METHODS:
            allocations[(alpha, method)] = integer_allocation(main_total, weights[method], minimum)
    maximum = np.max(np.vstack(list(allocations.values())), axis=0)

    draws = yield [(g, int(n)) for g, n in zip(groups, maximum)]
    streams = dict(zip(groups, draws))
    rows = []
    for (alpha, method), counts in allocations.items():
        empirical = counts_from_streams(model, streams, _mapping(model, counts))
        rows.append((alpha, method, float(solve_stop_loss(model_from_counts(model, empirical), alpha).cvar)))

    runs = total // H
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    counts = {g: np.zeros(model.kernels[g].size, dtype=np.int64) for g in groups}
    for step in range(H):
        h = H - step
        current = [int(s) for s in np.unique(state)]
        draws = yield [((h, s), int((state == s).sum())) for s in current]
        successor = np.empty_like(state)
        for s, d in zip(current, draws):
            mask = state == s
            kernel = model.kernels[(h, s)]
            totals[mask] += kernel.rewards[d]
            successor[mask] = kernel.next_states[d]
            counts[(h, s)] += np.bincount(d, minlength=kernel.size)
        state = successor
    # the same runs under the paper's estimator; a state no run reached is unreachable in the fitted model
    filled = {g: (c if c.sum() > 0 else np.ones_like(c)) for g, c in counts.items()}
    refit = model_from_counts(model, filled)
    for alpha in ALPHAS:
        rows.append((alpha, "complete_rollout", float(_empirical_cvar(totals, alpha))))
        rows.append((alpha, "rollout_model", float(solve_stop_loss(refit, alpha).cvar)))
    return rows


def drive(tasks: dict, sampler) -> dict:
    """Advance all tasks wave by wave; one sampler call per wave. tasks: key (environment, ...) -> generator."""
    results, requests = {}, {}
    for key, generator in tasks.items():
        requests[key] = next(generator)
    while requests:
        flat = [(key[0], g, n) for key, reqs in requests.items() for g, n in reqs]
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


# ----------------------------------------------------------------------------- samplers
class LawSampler:
    """Orders drawn from a decision law, then the simulator (equivalently: the exact kernel)."""

    def __init__(self, kernels: list, rng: np.random.Generator):
        self.kernels, self.rng, self.queries = kernels, rng, 0

    def draw(self, requests):
        self.queries += sum(n for *_, n in requests)
        return [self.rng.choice(len(self.kernels[e][g]), size=n, p=self.kernels[e][g]) for e, g, n in requests]


class EngineSampler:
    """Live: every order is one token sampled by vLLM from the model, restricted to the feasible digits."""

    def __init__(self, name: str, seed: int):
        from vllm import LLM, SamplingParams

        spec = CONFIG["models"][name]
        self.SamplingParams = SamplingParams
        self.temperature = float(spec["temperature"])
        self.top = int(spec.get("top_logprobs", 100))
        self.chunk = int(spec.get("samples_per_request", 4096))
        self.llm = LLM(model=spec["identifier"], revision=spec["revision"], tokenizer_revision=spec["revision"],
                       dtype=spec.get("dtype", "bfloat16"), seed=seed, enable_prefix_caching=True,
                       gpu_memory_utilization=float(spec.get("gpu_memory_utilization", 0.9)),
                       max_model_len=int(spec.get("max_model_len", 2048)),
                       max_num_seqs=int(spec.get("max_num_seqs", 1024)), max_logprobs=self.top,
                       **spec.get("engine_kwargs", {}))
        self.tokenizer = self.llm.get_tokenizer()
        self.digits = []
        for order in range(CAP + 1):
            encoded = self.tokenizer.encode(str(order), add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"digit {order} is not a single token")
            self.digits.append(int(encoded[0]))
        assert len(set(self.digits)) == CAP + 1
        self.prompts = {g: [int(t) for t in self.tokenizer.apply_chat_template(
            messages(*g), tokenize=True, add_generation_prompt=True, **spec.get("template_kwargs", {}))] for g in STATES}
        self.rng = np.random.default_rng([seed, 11])
        self.tables = [structure(p, loss)[1] for p, loss in ENVIRONMENTS]
        self.demand = {h: _censored_poisson_probabilities(INVENTORY_POISSON_MEAN[H - h]) for h in range(1, H + 1)}
        self.queries, self.seconds, self.logprob_modes = 0, 0.0, set()
        self.order_tally = {g: np.zeros(CAP - g[1] + 1, dtype=np.int64) for g in STATES}

    def allowed(self, group) -> list:
        return self.digits[:CAP - group[1] + 1]

    def orders(self, wanted: dict) -> dict:
        """wanted: group -> number of orders. One engine completion per order."""
        prompts, params, owner = [], [], []
        for group, n in wanted.items():
            allowed = self.allowed(group)
            for start in range(0, n, self.chunk):
                size = min(self.chunk, n - start)
                prompts.append({"prompt_token_ids": self.prompts[group]})
                params.append(self.SamplingParams(n=size, max_tokens=1, temperature=self.temperature, top_p=1.0,
                                                  allowed_token_ids=allowed, detokenize=False))
                owner.append(group)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        pieces = defaultdict(list)
        for group, output in zip(owner, outputs):
            index = {token: i for i, token in enumerate(self.allowed(group))}
            pieces[group].append(np.asarray([index[int(o.token_ids[0])] for o in output.outputs], dtype=int))
        result = {group: np.concatenate(parts) for group, parts in pieces.items()}
        for group, n in wanted.items():
            assert len(result[group]) == n
            self.order_tally[group] += np.bincount(result[group], minlength=CAP - group[1] + 1)
        return result

    def draw(self, requests):
        started = time.perf_counter()
        wanted = defaultdict(int)
        for _, group, n in requests:
            wanted[group] += n
        sampled = self.orders(dict(wanted))
        used = defaultdict(int)
        result = []
        for environment, group, n in requests:
            h, stock = group
            orders = sampled[group][used[group]:used[group] + n]
            used[group] += n
            demand = self.rng.choice(len(self.demand[h]), size=n, p=self.demand[h])
            p_base = ENVIRONMENTS[environment][0]
            disruption = np.where(stock + orders == CAP, p_base + 0.02, p_base)
            disrupted = (self.rng.random(n) < disruption).astype(int)
            result.append(self.tables[environment][group][0][orders, demand, disrupted])
        self.queries += sum(n for *_, n in requests)
        self.seconds += time.perf_counter() - started
        return result

    def law(self) -> dict:
        """Exact decision law in every state from the engine's log-probabilities (None if a digit is missing).

        The engine reports raw log-probabilities (before the restriction and the temperature; other tokens then
        have finite values) or processed ones (after both)."""
        params = [self.SamplingParams(n=1, max_tokens=1, temperature=self.temperature, top_p=1.0,
                                      allowed_token_ids=self.allowed(g), logprobs=self.top, detokenize=False) for g in STATES]
        outputs = self.llm.generate([{"prompt_token_ids": self.prompts[g]} for g in STATES], params, use_tqdm=False)
        laws = {}
        for group, output in zip(STATES, outputs):
            allowed = self.allowed(group)
            reported = output.outputs[0].logprobs[0]
            raw = any(np.isfinite(v.logprob) for t, v in reported.items() if t not in allowed)
            self.logprob_modes.add("raw" if raw else "processed")
            values = [reported[t].logprob if t in reported else None for t in allowed]
            if any(v is None or not np.isfinite(v) for v in values):
                laws[group] = None
                continue
            scaled = np.asarray(values, dtype=float) / (self.temperature if raw else 1.0)
            weights = np.exp(scaled - scaled.max())
            laws[group] = weights / weights.sum()
        return laws


def transformers_law(name: str) -> dict:
    """The same decision law through transformers: restricted softmax of the next-token logits."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = CONFIG["models"][name]
    tokenizer = AutoTokenizer.from_pretrained(spec["identifier"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(spec["identifier"], revision=spec["revision"],
                                                 torch_dtype=getattr(torch, spec.get("dtype", "bfloat16")), device_map="auto")
    model.eval()
    digits = [int(tokenizer.encode(str(k), add_special_tokens=False)[0]) for k in range(CAP + 1)]
    laws = {}
    with torch.inference_mode():
        for group in STATES:
            encoded = tokenizer.apply_chat_template(messages(*group), tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True, **spec.get("template_kwargs", {}))
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            logits = model(**encoded, use_cache=False).logits[0, -1, digits[:CAP - group[1] + 1]].float()
            laws[group] = torch.softmax(logits / float(spec["temperature"]), dim=0).double().cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return laws


# ----------------------------------------------------------------------------- commands
def environments_for(law: dict):
    """Per environment: (structure model, exact kernels, model with the exact kernels)."""
    out = []
    for p_base, loss in ENVIRONMENTS:
        model, tables = structure(p_base, loss)
        kernels = exact_kernels(model, tables, law)
        out.append((model, kernels, model.replace_query_probabilities(kernels)))
    return out


def write_policy(name: str, law: dict, extra: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, "laws": {f"{h}|{s}": [float(x) for x in law[(h, s)]] for h, s in STATES}, **extra}
    (OUT / f"policy_{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def synthetic(name: str, epsilon: float) -> None:
    from tis.environments import INVENTORY_BASE_STOCK

    law = {}
    for h, stock in STATES:
        p = np.full(CAP - stock + 1, epsilon / (CAP - stock + 1))
        p[min(CAP - stock, max(0, INVENTORY_BASE_STOCK[H - h] - stock))] += 1 - epsilon
        law[(h, stock)] = p
    write_policy(name, law, {"source": f"synthetic: base-stock with probability {1 - epsilon:g}, else a uniform order"})


def policy(name: str, through_transformers: bool) -> None:
    """Exact decision law. With --transformers: the law through transformers only (a separate process, so
    that the engine has the GPU to itself); otherwise the engine's law, compared with that file if present."""
    OUT.mkdir(parents=True, exist_ok=True)
    check_path = OUT / f"transformers_law_{name}.json"
    if through_transformers:
        law = transformers_law(name)
        check_path.write_text(json.dumps({f"{h}|{s}": [float(x) for x in law[(h, s)]] for h, s in STATES}) + "\n")
        print("wrote", check_path)
        return
    report = {}
    reference = None
    if check_path.exists():
        stored = json.loads(check_path.read_text())
        reference = {g: np.asarray(stored[f"{g[0]}|{g[1]}"], dtype=float) for g in STATES}
    sampler = EngineSampler(name, SEED + 5)
    law = sampler.law()
    report["logprob_modes"] = sorted(sampler.logprob_modes)
    missing = [g for g in STATES if law[g] is None]
    if missing:
        if reference is None:
            raise SystemExit(f"digits missing from the engine's log-probabilities at {missing}")
        report["target"] = "transformers (a digit was missing from the engine's log-probabilities)"
        law = reference
    else:
        report["target"] = "engine log-probabilities"
    if reference is not None:
        report["largest_gap_engine_to_transformers"] = max(
            float(np.abs(law[g] - reference[g]).max()) for g in STATES)
    write_policy(name, law, {"source": report["target"], "report": report,
                             "identifier": CONFIG["models"][name]["identifier"],
                             "revision": CONFIG["models"][name]["revision"],
                             "temperature": CONFIG["models"][name]["temperature"]})
    print(json.dumps(report, indent=1))
    show(name)


def show(name: str) -> None:
    law = read_law(name)
    print(f"decision probabilities of {name} (rows: period; columns: stock 0..6; most likely order and its probability)")
    for h in range(H, 0, -1):
        print(f"  period {H - h + 1}: " + "  ".join(f"{int(np.argmax(law[(h, s)]))}({law[(h, s)].max():.2f})" for s in range(CAP + 1)))


def predict(names) -> None:
    for name in names:
        law = read_law(name)
        print(f"{name}: first-order MSE relative to fitted-model rollouts at {BUDGET} queries per state "
              f"(median over the {len(ENVIRONMENTS)} environments [min, max])")
        for alpha in ALPHAS:
            rows = []
            for model, _, exact in environments_for(law):
                t = compute_influences(exact, alpha)
                s, mean = np.asarray(t.tail_scales), np.asarray(t.mean_scales)
                d = _occupancy_scores(exact, t.occupancy)
                G = exact.group_count
                N = G * BUDGET
                v_roll = float(np.sum(s[d > 0] ** 2 / ((N // H) * d[d > 0])))
                expl = N ** (-float(PILOT["exploration_exponent"]))
                m = pilot_size(N, G, float(PILOT["multiplier"]), int(PILOT["minimum_pilot_per_group"]), int(PILOT["minimum_main_per_group"]))
                w_occ = regularized_weights(d, expl)
                designs = {"tis": regularized_weights(s, expl),
                           "tis_anchored": regularized_weights(_anchored_score(s, d), expl),
                           "occ_plus_mean": 0.5 * regularized_weights(mean, expl) + 0.5 * w_occ,
                           "learned_occupancy": w_occ}
                row = {k: float(np.sum(s ** 2 / integer_allocation(N - G * m, w, int(PILOT["minimum_main_per_group"])))) / v_roll
                       for k, w in designs.items()}
                row["cvar_profit"] = B.SPAN * float(t.solution.cvar) + H * B.LOW
                row["margin"] = float(t.solution.margin)
                rows.append(row)
            line = "  ".join(f"{k} {np.median([r[k] for r in rows]):.2f} [{min(r[k] for r in rows):.2f}, {max(r[k] for r in rows):.2f}]"
                             for k in ("tis", "tis_anchored", "occ_plus_mean", "learned_occupancy"))
            print(f"  alpha {alpha:g}: {line}; CVaR profit {min(r['cvar_profit'] for r in rows):.2f} to "
                  f"{max(r['cvar_profit'] for r in rows):.2f}; smallest margin {min(r['margin'] for r in rows):.5f}")


def replication_rows(name: str, replication: int, results: dict, truths: list) -> list:
    rows = []
    for (environment, _), task_rows in results.items():
        for alpha, method, estimate in task_rows:
            rows.append({"model": name, "environment": environment, "replication": replication, "alpha": alpha,
                         "method": method, "error_profit": repr(B.SPAN * (estimate - truths[environment][alpha]))})
    return rows


def _simulate_job(args):
    name, replication = args
    built = environments_for(read_law(name))
    truths = [{a: float(solve_stop_loss(exact, a).cvar) for a in ALPHAS} for _, _, exact in built]
    sampler = LawSampler([kernels for _, kernels, _ in built], np.random.default_rng([SEED, 3, replication]))
    results = drive({(e, replication): task(model) for e, (model, _, _) in enumerate(built)}, sampler)
    return replication_rows(name, replication, results, truths)


def simulate(names, replications: int, processes: int) -> None:
    from multiprocessing import Pool

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [(name, r) for name in names for r in range(replications)]
    with Pool(processes) as pool, (OUT / "raw_simulated.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for done, rows in enumerate(pool.imap_unordered(_simulate_job, jobs), 1):
            writer.writerows(rows)
            stream.flush()
            if done % 10 == 0:
                print(f"{done} of {len(jobs)} replications, {time.time() - started:.0f}s", flush=True)


def _binomial_p(count: int, n: int, p: float) -> float:
    from tis.finqa.calibrate import _binomial_two_sided_p  # the exact test of the paper's generation audit

    return float(_binomial_two_sided_p(int(count), int(n), float(min(max(p, 0.0), 1.0))))


def orders_against_law(tally: dict, law: dict) -> dict:
    """Sampled order frequencies against the decision law: exact binomial test of every cell whose gap
    exceeds three standard errors, Bonferroni over all cells at family level .01."""
    cells, largest_z, exact = 0, 0.0, []
    for group in STATES:
        n = int(tally[group].sum())
        if n == 0 or len(law[group]) == 1:
            continue
        p = np.asarray(law[group])
        z = (tally[group] - n * p) / np.sqrt(np.maximum(n * p * (1 - p), 1e-300))
        cells += len(p)
        largest_z = max(largest_z, float(np.abs(z).max()))
        for j in np.flatnonzero(np.abs(z) > 3):
            exact.append([list(group), int(j), int(tally[group][j]), n, float(p[j]), _binomial_p(tally[group][j], n, p[j])])
    threshold = 0.01 / max(cells, 1)
    return {"cells": cells, "orders": int(sum(int(t.sum()) for t in tally.values())),
            "largest_standardized_gap": largest_z, "cells_tested_exactly": len(exact),
            "bonferroni_threshold_family_.01": threshold,
            "cells_below_threshold": [e for e in exact if e[-1] < threshold]}


def check(name: str, samples: int) -> None:
    law = read_law(name)
    sampler = EngineSampler(name, SEED + 6)
    started = time.perf_counter()
    sampler.orders({g: samples for g in STATES})
    elapsed = time.perf_counter() - started
    report = {"orders_against_law": orders_against_law(sampler.order_tally, law),
              "throughput": {"orders": samples * len(STATES), "seconds": elapsed,
                             "orders_per_second": samples * len(STATES) / elapsed}}
    built = environments_for(law)
    started, before = time.perf_counter(), sampler.queries
    drive({(e, 0): task(model) for e, (model, _, _) in enumerate(built)}, sampler)
    report["one_replication_of_all_environments"] = {"seconds": time.perf_counter() - started,
                                                     "queries": sampler.queries - before}
    (OUT / f"check_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def run(name: str, shard: int, shards: int, together: int) -> None:
    raw = OUT / f"raw_{name}_shard{shard}of{shards}.csv"
    tally_path = OUT / f"orders_{name}_shard{shard}of{shards}.json"
    done = {int(r["replication"]) for r in csv.DictReader(raw.open())} if raw.exists() else set()
    built = environments_for(read_law(name))
    truths = [{a: float(solve_stop_loss(exact, a).cvar) for a in ALPHAS} for _, _, exact in built]
    sampler = EngineSampler(name, SEED + 1000 * shards + shard)
    if tally_path.exists():
        stored = json.loads(tally_path.read_text())
        for g in STATES:
            sampler.order_tally[g] += np.asarray(stored[f"{g[0]}|{g[1]}"], dtype=np.int64)
    todo = [r for r in range(int(CONFIG["replications"])) if r % shards == shard and r not in done]
    for start in range(0, len(todo), together):
        batch = todo[start:start + together]
        started = time.perf_counter()
        results = drive({(e, r): task(model) for r in batch for e, (model, _, _) in enumerate(built)}, sampler)
        rows = []
        for r in batch:
            rows += replication_rows(name, r, {k: v for k, v in results.items() if k[1] == r}, truths)
        fresh = not raw.exists()
        with raw.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if fresh:
                writer.writeheader()
            writer.writerows(rows)
        tally_path.write_text(json.dumps({f"{g[0]}|{g[1]}": sampler.order_tally[g].tolist() for g in STATES}) + "\n")
        print(f"replications {batch[0]}-{batch[-1]}: {time.perf_counter() - started:.0f}s, engine {sampler.seconds:.0f}s, "
              f"cumulative queries {sampler.queries}", flush=True)


def family(sq: dict, a: str, b: str, paired: bool, rng, resamples: int) -> dict:
    """Ratio of equal-environment-weight MSEs; bootstrap over replications within environment; centred test.
    Also the environments in which a is resolved lower or higher (|z| >= 2)."""
    environments = sorted({e for e, _ in sq})
    num = np.mean([sq[(e, a)].mean() for e in environments])
    den = np.mean([sq[(e, b)].mean() for e in environments])
    ba, bb, lower, higher, ratios = 0.0, 0.0, 0, 0, []
    for e in environments:
        sa, sb = sq[(e, a)], sq[(e, b)]
        ia = rng.integers(0, sa.size, size=(resamples, sa.size))
        ib = ia if paired else rng.integers(0, sb.size, size=(resamples, sb.size))
        ba = ba + sa[ia].mean(axis=1) / len(environments)
        bb = bb + sb[ib].mean(axis=1) / len(environments)
        if paired:
            d = sa - sb
            z = d.mean() / (d.std(ddof=1) / np.sqrt(d.size)) if d.std() > 0 else 0.0
        else:
            z = (sa.mean() - sb.mean()) / np.sqrt(sa.var(ddof=1) / sa.size + sb.var(ddof=1) / sb.size)
        lower += int(z <= -2)
        higher += int(z >= 2)
        ratios.append(float(sa.mean() / sb.mean()))
    centred = (ba - bb) - (num - den)
    return {"mse_ratio": float(num / den), "ratio_ci95": [float(np.quantile(ba / bb, .025)), float(np.quantile(ba / bb, .975))],
            "bootstrap_p_two_sided": float(np.mean(np.abs(centred) >= abs(num - den))),
            "environments_lower": lower, "environments_higher": higher,
            "environment_ratios": [round(r, 3) for r in ratios]}


def load(simulated: bool) -> dict:
    paths = [OUT / "raw_simulated.csv"] if simulated else sorted(OUT.glob("raw_*_shard*.csv"))
    errors = defaultdict(dict)
    for path in paths:
        for r in csv.DictReader(path.open()):
            errors[(r["model"], float(r["alpha"]), int(r["environment"]), r["method"])][int(r["replication"])] = float(r["error_profit"])
    return errors


def analyze(simulated: bool) -> None:
    errors = load(simulated)
    rng = np.random.default_rng(SEED + 1)
    resamples = int(CONFIG["bootstrap_resamples"])
    stream = set(STREAM_METHODS)
    analysis = {"primary": {}, "secondary": {}, "replications": {}}
    names = sorted({k[0] for k in errors})

    def squares(name, alpha):
        return {(e, m): np.array([v[r] ** 2 for r in sorted(v)]) for (n, a, e, m), v in errors.items() if n == name and a == alpha}

    for name in names:
        for alpha in ALPHAS:
            sq = squares(name, alpha)
            analysis["replications"][f"{name}|{alpha:g}"] = sorted({v.size for v in sq.values()})
            for a, b in CONFIG["secondary"]:
                paired = (a in stream) == (b in stream)
                analysis["secondary"][f"{name}|{alpha:g}|{a}_vs_{b}"] = family(sq, a, b, paired, rng, resamples)
    for label, p in CONFIG["primary"].items():
        if p["model"] in names:
            analysis["primary"][label] = family(squares(p["model"], float(p["alpha"])), p["method"], p["reference"],
                                                False, rng, resamples)
    ps = {k: max(v["bootstrap_p_two_sided"], 1.0 / resamples) for k, v in analysis["primary"].items()}
    adjusted, running = {}, 0.0
    for rank, key in enumerate(sorted(ps, key=ps.get)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[key]))
        adjusted[key] = running
    analysis["holm"] = {"raw": ps, "adjusted": adjusted}
    if not simulated:
        for name in names:
            tally = {g: np.zeros(CAP - g[1] + 1, dtype=np.int64) for g in STATES}
            for path in sorted(OUT.glob(f"orders_{name}_shard*.json")):
                stored = json.loads(path.read_text())
                for g in STATES:
                    tally[g] += np.asarray(stored[f"{g[0]}|{g[1]}"], dtype=np.int64)
            analysis.setdefault("orders_against_law", {})[name] = orders_against_law(tally, read_law(name))
    (OUT / ("analysis_simulated.json" if simulated else "analysis.json")).write_text(json.dumps(analysis, indent=1) + "\n")
    print(json.dumps({"replications": analysis["replications"], "holm": analysis["holm"]}, indent=1))
    for key, v in {**analysis["primary"], **analysis["secondary"]}.items():
        print(f"  {key}: {v['mse_ratio']:.3f} [{v['ratio_ci95'][0]:.3f}, {v['ratio_ci95'][1]:.3f}] p {v['bootstrap_p_two_sided']:.4f} "
              f"(environments lower {v['environments_lower']}, higher {v['environments_higher']})")
    if "orders_against_law" in analysis:
        print(json.dumps(analysis["orders_against_law"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prompts")
    s = sub.add_parser("synthetic")
    s.add_argument("--name", required=True)
    s.add_argument("--epsilon", type=float, required=True)
    p = sub.add_parser("policy")
    p.add_argument("--model", required=True)
    p.add_argument("--transformers", action="store_true")
    q = sub.add_parser("predict")
    q.add_argument("--models", nargs="+", default=list(CONFIG["models"]))
    m = sub.add_parser("simulate")
    m.add_argument("--models", nargs="+", default=list(CONFIG["models"]))
    m.add_argument("--replications", type=int, default=int(CONFIG["replications"]))
    m.add_argument("--processes", type=int, default=3)
    c = sub.add_parser("check")
    c.add_argument("--model", required=True)
    c.add_argument("--samples", type=int, default=20000)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--together", type=int, default=4)
    a = sub.add_parser("analyze")
    a.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    if args.command == "prompts":
        for group in ((H, 0), (4, 3), (1, 6)):
            print(json.dumps(messages(*group), indent=1))
    elif args.command == "synthetic":
        synthetic(args.name, args.epsilon)
    elif args.command == "policy":
        policy(args.model, args.transformers)
    elif args.command == "predict":
        predict(args.models)
    elif args.command == "simulate":
        simulate(args.models, args.replications, args.processes)
    elif args.command == "check":
        check(args.model, args.samples)
    elif args.command == "run":
        k, n = (int(x) for x in args.shard.split("/"))
        run(args.model, k, n, args.together)
    else:
        analyze(args.simulated)


if __name__ == "__main__":
    main()
