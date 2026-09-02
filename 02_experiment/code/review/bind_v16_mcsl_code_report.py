from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
STATE = ROOT / "00_project" / "researchpilot_state.json"
REPORT = ROOT / "02_experiment" / "code" / "review" / "CODE_REPORT_V16_MCSL_R1.json"

sys.path.insert(0, r"C:\Users\Administrator\.codex\skills\researchpilot\scripts")
from state_lock import StateLock, atomic_write_text  # noqa: E402


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    if report.get("status") != "PASS" or report.get("route_id") != "R-EO-MCSL-V16-01":
        raise RuntimeError("V16 MCSL code report is not passing")
    with StateLock(ROOT):
        state = json.loads(STATE.read_text(encoding="utf-8-sig"))
        state["active_phase"] = "EXPERIMENT"
        state["active_service"] = "experiment"
        state["active_gate"] = "CLOUD_SYNC"
        state["experiment_status"] = "v16_mcsl_local_code_pass_cloud_sync_pending"
        state["child_state_refs"]["code"] = "02_experiment/code/review/CODE_REPORT_V16_MCSL_R1.json"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(STATE, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "pass", "active_gate": "CLOUD_SYNC", "code_ref": "02_experiment/code/review/CODE_REPORT_V16_MCSL_R1.json"}, indent=2))


if __name__ == "__main__":
    main()
