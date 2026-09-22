"""Screened FinQA panel construction (protocol v2): stub-generator tests.

unittest-style, so the documented command
`PYTHONPATH=src python -m unittest discover -s tests` collects every test.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tis.finqa.candidates import GENERATION_SEED_BASE, build_screened_panel
from tis.finqa.panel import _seeded_rank, normalize_items

PROTOCOL = Path(__file__).resolve().parent.parent / "configs" / "finqa_protocol.json"


def _write_source(path: Path, count: int) -> None:
    rows = []
    for i in range(count):
        rows.append(
            {
                "id": f"CO/2020/page_{i:02d}.pdf-1",
                "qa": {
                    "exe_ans": 100.0 + i,
                    "question": f"what was value {i}?",
                    "program": "",
                },
                "pre_text": ["Revenue grew."],
                "post_text": [],
                "table": [["year", "value"], ["2020", str(100 + i)]],
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _informative(item_id: str) -> bool:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 3 == 0


def _stub_generator(calls: list):
    def generate(item, item_seed_base):
        calls.append((item["item_id"], item_seed_base))
        gold = float(item["gold_answer"])
        answers = [gold] * 8
        if _informative(item["item_id"]):
            answers[3] = 3.0 * gold  # severe outcome: utility 1/3 vs 1.0
        candidates = [
            {
                "slot": slot,
                "derivation": "stub",
                "raw_answer": answer,
                "is_percent": False,
                "answer": answer,
                "valid": True,
                "output_tokens": 5,
            }
            for slot, answer in enumerate(answers)
        ]
        return {
            "prompt_tokens": 10,
            "candidates": candidates,
            "input_tokens": 80,
            "output_tokens": 40,
            "wall_seconds": 0.01,
        }

    return generate


def _build(tmp_path: Path, source: Path, tag: str, size: int, seed: int,
           exclude=(), generator=None):
    return build_screened_panel(
        PROTOCOL,
        source,
        tmp_path / f"panel_{tag}.jsonl",
        tmp_path / f"manifest_{tag}.json",
        tmp_path / f"banks_{tag}.json",
        tmp_path / f"scan_{tag}.jsonl",
        size,
        seed,
        0.25,
        exclude_item_ids=exclude,
        generator=generator,
    )


class FinqaScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_screen_admits_only_spread_and_seeds_by_scan_rank(self):
        tmp_path = self.tmp_path
        source = tmp_path / "source.json"
        _write_source(source, 30)
        calls: list = []
        manifest = _build(tmp_path, source, "dev", 5, 1234,
                          generator=_stub_generator(calls))

        items = normalize_items(source)
        ordered = [it["item_id"] for it in
                   sorted(items, key=lambda it: _seeded_rank(1234, it["item_id"]))]
        self.assertEqual(manifest["scanned_item_ids"], ordered[: manifest["scanned"]])
        self.assertEqual(calls, [
            (item_id, GENERATION_SEED_BASE + 1000 * rank)
            for rank, item_id in enumerate(manifest["scanned_item_ids"])
        ])
        self.assertTrue(all(_informative(i) for i in manifest["item_ids"]))
        self.assertEqual(len(manifest["item_ids"]), 5)
        accepted_in_scan = [
            i for i, flag in zip(manifest["scanned_item_ids"],
                                 manifest["accepted_flags"]) if flag
        ]
        self.assertEqual(sorted(accepted_in_scan), manifest["item_ids"])

        panel = [json.loads(l) for l in
                 (tmp_path / "panel_dev.jsonl").read_text().splitlines()]
        self.assertEqual([p["item_id"] for p in panel],
                         sorted(p["item_id"] for p in panel))
        self.assertEqual([p["panel_index"] for p in panel], list(range(5)))
        banks = json.loads((tmp_path / "banks_dev.json").read_text())["banks"]
        self.assertEqual([(b["panel_index"], b["item_id"]) for b in banks],
                         [(p["panel_index"], p["item_id"]) for p in panel])
        scan = [json.loads(l) for l in
                (tmp_path / "scan_dev.jsonl").read_text().splitlines()]
        for record in scan:
            self.assertEqual(record["accepted"],
                             record["spread"] >= 0.25 - 1e-12)

    def test_resume_replays_scan_without_regeneration(self):
        tmp_path = self.tmp_path
        source = tmp_path / "source.json"
        _write_source(source, 30)
        first = _build(tmp_path, source, "dev", 5, 1234,
                       generator=_stub_generator([]))

        def exploding(item, seed):
            raise AssertionError("resume must not regenerate scanned banks")

        second = _build(tmp_path, source, "dev", 5, 1234, generator=exploding)
        self.assertEqual(second, first)

    def test_scan_file_from_other_selection_is_rejected(self):
        tmp_path = self.tmp_path
        source = tmp_path / "source.json"
        _write_source(source, 30)
        _build(tmp_path, source, "dev", 5, 1234, generator=_stub_generator([]))
        (tmp_path / "scan_other.jsonl").write_text(
            (tmp_path / "scan_dev.jsonl").read_text())
        with self.assertRaisesRegex(ValueError, "different selection"):
            build_screened_panel(
                PROTOCOL, source,
                tmp_path / "p.jsonl", tmp_path / "m.json", tmp_path / "b.json",
                tmp_path / "scan_other.jsonl",
                5, 9999, 0.25, generator=_stub_generator([]),
            )

    def test_heldout_excludes_all_scanned_dev_items(self):
        tmp_path = self.tmp_path
        source = tmp_path / "source.json"
        _write_source(source, 60)
        dev = _build(tmp_path, source, "dev", 5, 1234,
                     generator=_stub_generator([]))
        heldout = _build(
            tmp_path, source, "heldout", 5, 5678,
            exclude=dev["scanned_item_ids"], generator=_stub_generator([]),
        )
        self.assertFalse(set(heldout["scanned_item_ids"]) &
                         set(dev["scanned_item_ids"]))

    def test_exhausted_pool_raises(self):
        tmp_path = self.tmp_path
        source = tmp_path / "source.json"
        _write_source(source, 6)
        with self.assertRaisesRegex(ValueError, "admitted only"):
            _build(tmp_path, source, "dev", 6, 1234,
                   generator=_stub_generator([]))

    def test_binomial_p_value_survives_large_sample_counts(self):
        from tis.finqa.calibrate import _binomial_two_sided_p

        # n=2048 overflows a naive comb()-based enumeration.
        self.assertTrue(0.0 < _binomial_two_sided_p(1024, 2048, 0.5) <= 1.0)
        self.assertAlmostEqual(_binomial_two_sided_p(0, 2048, 1e-9), 1.0,
                               delta=1e-5)
        self.assertLess(_binomial_two_sided_p(2048, 2048, 1e-9), 1e-300)
        # degenerate enumerated probabilities (float underflow) score directly
        self.assertEqual(_binomial_two_sided_p(0, 2048, 0.0), 1.0)
        self.assertEqual(_binomial_two_sided_p(3, 2048, 0.0), 0.0)
        self.assertEqual(_binomial_two_sided_p(2048, 2048, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
