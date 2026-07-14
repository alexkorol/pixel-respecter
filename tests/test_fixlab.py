from pathlib import Path

import numpy as np

from benchmarks.eval_wild import exact_cell_accuracy
from scripts import fixes_to_tests


ROOT = Path(__file__).resolve().parent.parent


def test_exact_cell_accuracy_is_strict_and_penalizes_shape():
    truth = np.zeros((2, 2, 3), dtype=np.uint8)
    result = truth.copy()
    assert exact_cell_accuracy(result, truth) == 100.0

    result[0, 0, 0] = 1
    assert exact_cell_accuracy(result, truth) == 75.0

    smaller = np.zeros((1, 2, 3), dtype=np.uint8)
    assert exact_cell_accuracy(smaller, truth) == 50.0

    transparent_a = np.array([[[255, 0, 0, 0]]], dtype=np.uint8)
    transparent_b = np.array([[[0, 255, 0, 0]]], dtype=np.uint8)
    assert exact_cell_accuracy(transparent_a, transparent_b) == 100.0
    assert exact_cell_accuracy(transparent_a, np.zeros((1, 1, 3), dtype=np.uint8)) == 0.0


def test_fixlab_has_human_comparison_and_native_preview():
    page = (ROOT / "fixlab" / "index.html").read_text(encoding="utf-8")
    assert 'id="sourceCanvas"' in page
    assert 'id="artCanvas"' in page
    assert 'id="nativeCanvas"' in page
    assert "Pixel-perfected · enlarged" in page
    assert "Actual 1× export" in page
    assert 'id="showGrid"' in page
    assert 'id="autoTransparency"' in page
    assert 'id="autoCrop"' in page
    assert 'id="cropPadding"' in page
    assert 'id="saveNext"' in page
    assert 'id="exportSession"' in page


def test_generated_regressions_skip_incomplete_calibrations(tmp_path, monkeypatch):
    corpus = tmp_path / "wild_corpus"
    fixes = corpus / "fixes"
    truth = corpus / "ground_truth"
    fixes.mkdir(parents=True)
    truth.mkdir()
    (fixes / "complete.jsonl").write_text(
        '{"id":"complete","kind":"freehand","context":{"cell":[1,2]},'
        '"choice":{"color":[3,4,5]}}\n',
        encoding="utf-8",
    )
    (fixes / "in_progress.jsonl").write_text(
        '{"id":"in_progress","kind":"freehand","context":{"cell":[0,0]},'
        '"choice":{"color":[9,9,9]}}\n',
        encoding="utf-8",
    )
    (truth / "complete.png").write_bytes(b"completed")

    monkeypatch.setattr(fixes_to_tests, "CORPUS", corpus)

    cases = fixes_to_tests.build_cases()
    assert [case["id"] for case in cases] == ["complete"]
