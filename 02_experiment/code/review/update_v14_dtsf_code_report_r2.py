from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
REPORT = ROOT / "02_experiment" / "code" / "review" / "CODE_REPORT_V14_DTSF_R2.json"
MANIFEST = ROOT / "02_experiment" / "code" / "manifests" / "clean_sync_manifest_v14_dtsf_20260830_r4.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / "geotoken3path_code_v14_dtsf_20260830_r4.tar.gz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report["clean_sync_manifest_sha256"] = sha(MANIFEST)
    report["packaging_closure"]["clean_sync_manifest"] = {"path": str(MANIFEST), "sha256": sha(MANIFEST), "file_count": manifest["file_count"]}
    report["packaging_closure"]["release_package"] = {"path": str(PACKAGE), "sha256": sha(PACKAGE), "bytes": PACKAGE.stat().st_size, "member_count": len(__import__("tarfile").open(PACKAGE, "r:gz").getnames())}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "manifest_sha256": sha(MANIFEST), "package_sha256": sha(PACKAGE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
