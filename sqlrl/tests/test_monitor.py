import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor

ARGS = SimpleNamespace(collapse=0.15, hang_minutes=45, mem_alert=24)


def metric(step, **kw):
    return {"step": step, "peak_mem_gb": 6.0, "reward_mean": 0.5, **kw}


def test_literal_correct_answer_stops_run(tmp_path):
    (tmp_path / "samples").mkdir()
    sample = {"source": "student", "outcome": "correct", "literal": True}
    (tmp_path / "samples" / "step_00003.json").write_text(json.dumps([sample]))
    alerts, stop = monitor.check(tmp_path, [metric(3)], [], None, ARGS, 0)
    assert "literal" in stop
    sample["source"] = "teacher"
    (tmp_path / "samples" / "step_00003.json").write_text(json.dumps([sample]))
    assert monitor.check(tmp_path, [metric(3)], [], None, ARGS, 0)[1] is None


def test_eval_collapse_needs_two_low_evaluations(tmp_path):
    evals = [{"step": s, "pass1": p} for s, p in [(0, 0.5), (10, 0.6), (20, 0.3)]]
    assert monitor.check(tmp_path, [], evals, None, ARGS, 0)[1] is None
    evals.append({"step": 30, "pass1": 0.32})
    assert "collapse" in monitor.check(tmp_path, [], evals, None, ARGS, 0)[1]


def test_hang_and_alerts(tmp_path):
    status = {"phase": "rollout", "step": 4, "updated": 0}
    assert "no status update" in monitor.check(tmp_path, [], [], status, ARGS, 46 * 60)[1]
    alerts, _ = monitor.check(tmp_path, [metric(4, update_skipped_nonfinite=True, peak_mem_gb=25)], [], None, ARGS, 0)
    assert len(alerts) == 2
