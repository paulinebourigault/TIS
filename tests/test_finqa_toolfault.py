"""Tool-fault FinQA model: structure and reduction to the base workflow."""
from __future__ import annotations

import unittest

import numpy as np

from tis.engine import compute_influences
from tis.finqa.models import finqa_question_model
from tis.finqa.toolfault import (
    STATES_TF, base_state, render_tf_messages, toolfault_question_model,
)


def _bank():
    answers = [10.0, 10.0, 1000.0, -10.0, 10.0, 11.0, 10.0, 0.1]
    return [{"slot": i, "derivation": "d", "raw_answer": a, "is_percent": False,
             "answer": a, "valid": True, "output_tokens": 1}
            for i, a in enumerate(answers)]


def _question(rng, states):
    kernels = []
    for s in range(states):
        law = rng.dirichlet(np.ones(72))
        kernels.append({"state_index": s, "action": "x", "probabilities": law.tolist()})
    return {"panel_index": 0, "item_id": "t", "gold_answer": 10.0, "kernels": kernels}


class ToolFaultModelTests(unittest.TestCase):
    def test_kernel_laws_sum_to_one_and_fault_mass(self):
        q = _question(np.random.default_rng(0), STATES_TF)
        m = toolfault_question_model(q, _bank(), 3, 0.05)
        self.assertEqual(m.group_count, STATES_TF)
        for (state, _), k in m.kernels.items():
            self.assertAlmostEqual(float(k.probabilities.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(k.probabilities[72:].sum()), 0.05, places=12)
            self.assertTrue(np.all(k.next_states[72:] > 72))

    def test_reduces_to_base_when_fault_states_copy_base_laws(self):
        rng = np.random.default_rng(1)
        base_q = _question(rng, 73)
        laws = {k["state_index"]: k["probabilities"] for k in base_q["kernels"]}
        tf_q = dict(base_q)
        tf_q["kernels"] = [{"state_index": s, "action": "x",
                            "probabilities": laws[base_state(s)]} for s in range(STATES_TF)]
        bank = _bank()
        base = compute_influences(finqa_question_model(base_q, bank, 3), 0.1).solution.cvar
        for p in (0.01, 0.3):
            tf = compute_influences(toolfault_question_model(tf_q, bank, 3, p), 0.1).solution.cvar
            self.assertAlmostEqual(tf, base, places=12)

    def test_prompts_differ_only_in_reported_value(self):
        item = {"context": "ctx", "question": "q?"}
        bank = _bank()
        normal = render_tf_messages(item, bank, 1 + 9 * 2 + 4, "ordinary")[1]["content"]
        faulted = render_tf_messages(item, bank, 1 + 72 + 9 * 2 + 4, "ordinary")[1]["content"]
        self.assertIn("returned: 1000.", normal)
        self.assertIn("returned: 100000.", faulted)
        self.assertEqual(normal.replace("returned: 1000.", "returned: X."),
                         faulted.replace("returned: 100000.", "returned: X."))
        root = render_tf_messages(item, bank, 0, "ordinary")[1]["content"]
        self.assertNotIn("calculator", root)


if __name__ == "__main__":
    unittest.main()
