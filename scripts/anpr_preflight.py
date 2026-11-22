"""Read-only readiness check for the local ANPR demonstration.

Run from anywhere with the project interpreter:

    .venv/bin/python scripts/anpr_preflight.py            # RapidOCR demo, MPS detector
    .venv/bin/python scripts/anpr_preflight.py --engine easyocr
    .venv/bin/python scripts/anpr_preflight.py --device cpu --json

The check never loads a model, downloads anything, starts a server, writes
files, or opens dataset images. It hashes local model files, compares
installed package versions with ``requirements-anpr.txt``, reads the prepared
dataset's metadata JSON, queries device availability and probes whether
local ports already have listeners.

Status meanings:
  PASS  the item matches the pinned/expected state;
  WARN  the demo can still run, but something differs from the recorded state;
  FAIL  the selected launcher configuration is expected to return UNAVAILABLE
        or fail to start.
Exit code is 1 when any FAIL is present, otherwise 0.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# The generic COCO base is not used at inference time; it is the pinned
# starting checkpoint recorded for detector training reproducibility.
YOLOV8N_BASE = ("models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36")
EASYOCR_ORIGINAL_FILES = ("models/easyocr/craft_mlt_25k.pth", "models/easyocr/english_g2.pth")
DATASET_DIR = "data/processed/witvix_anpr_pilot_v1"
DATASET_FINGERPRINT = "2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a"
DATASET_SPLIT_IMAGES = {"train": 329, "val": 71, "test": 70}
NEXT_SWC_BINARY = "apps/web/node_modules/@next/swc-darwin-arm64/next-swc.darwin-arm64.node"
# Packages whose absence prevents the selected engine from starting.
REQUIRED_FOR = {
    "common": ("torch", "torchvision", "ultralytics", "numpy", "opencv-python-headless",
               "fastapi", "uvicorn", "pydantic"),
    "rapidocr": ("rapidocr-onnxruntime", "onnxruntime"),
    "easyocr": ("easyocr",),
}


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class AssetSpec:
    relative_path: str
    sha256: str
    required: bool
    purpose: str


# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)")


def parse_pins(text: str) -> Dict[str, str]:
    """Return ``{canonical-name: version}`` for ``name==version`` lines only."""
    pins: Dict[str, str] = {}
    for line in text.splitlines():
        match = _PIN.match(line)
        if match:
            pins[canonical(match.group(1))] = match.group(2)
    return pins


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_launcher(root: Path):
    path = root / "scripts" / "serve_anpr_local.py"
    spec = importlib.util.spec_from_file_location("_roadsense_serve_anpr_local", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # guarded by __main__; defines constants only
    return module


def pinned_assets(engine: str, root: Path = ROOT) -> List[AssetSpec]:
    """Single-source the pins from the launcher and the RapidOCR adapter."""
    launcher = _load_launcher(root)
    defaults = launcher.DEFAULTS
    specs = [
        AssetSpec(defaults["ROADSENSE_ANPR_PLATE_WEIGHTS"], defaults["ROADSENSE_ANPR_PLATE_SHA256"],
                  True, "trained plate detector"),
        AssetSpec(YOLOV8N_BASE[0], YOLOV8N_BASE[1], False, "detector training base (not used at inference)"),
    ]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.anpr.ocr import RAPIDOCR_EXPECTED_SHA256  # numpy-only import, no model load

    for name, digest in RAPIDOCR_EXPECTED_SHA256.items():
        specs.append(AssetSpec(f"models/rapidocr/{name}", digest, engine == "rapidocr", "RapidOCR PP-OCRv4"))
    specs.append(AssetSpec(defaults["ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS"],
                           defaults["ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256"],
                           engine == "easyocr", "adapted EasyOCR recognizer"))
    return specs


# ── checks ───────────────────────────────────────────────────────────────────

def check_interpreter(root: Path, prefix: str, version: Sequence[int]) -> List[Check]:
    checks = []
    venv = (root / ".venv").resolve()
    inside = Path(prefix).resolve() == venv
    checks.append(Check("python", "interpreter", PASS if inside else WARN,
                        f"sys.prefix={prefix}" + ("" if inside else f" (expected {venv}; use .venv/bin/python)")))
    wanted_file = root / ".python-version"
    running = f"{version[0]}.{version[1]}.{version[2]}"
    if wanted_file.is_file():
        wanted = wanted_file.read_text().strip()
        same = running == wanted or running.startswith(wanted + ".")
        checks.append(Check("python", "version", PASS if same else WARN,
                            f"running {running}; .python-version {wanted}"))
    else:
        checks.append(Check("python", "version", WARN, f"running {running}; no .python-version file"))
    return checks


def check_packages(pins: Mapping[str, str], engine: str,
                   lookup: Callable[[str], Optional[str]]) -> List[Check]:
    required = set(REQUIRED_FOR["common"]) | set(REQUIRED_FOR[engine])
    checks = []
    for name in sorted(pins):
        installed = lookup(name)
        expected = pins[name]
        if installed is None:
            status = FAIL if name in required else WARN
            checks.append(Check("packages", name, status, f"not installed (recorded {expected})"))
        elif installed != expected:
            checks.append(Check("packages", name, WARN, f"installed {installed}; recorded {expected}"))
        else:
            checks.append(Check("packages", name, PASS, installed))
    for name in sorted(required - set(pins)):
        installed = lookup(name)
        checks.append(Check("packages", name, PASS if installed else FAIL,
                            installed or "not installed and not recorded in requirements-anpr.txt"))
    return checks


def check_assets(root: Path, specs: Iterable[AssetSpec],
                 hasher: Callable[[Path], str] = sha256_file) -> List[Check]:
    checks = []
    for spec in specs:
        path = root / spec.relative_path
        missing_status = FAIL if spec.required else WARN
        if path.is_symlink():
            checks.append(Check("assets", spec.relative_path, missing_status, "symlink refused by runtime"))
        elif not path.is_file():
            checks.append(Check("assets", spec.relative_path, missing_status, f"missing ({spec.purpose})"))
        else:
            actual = hasher(path)
            ok = actual == spec.sha256
            checks.append(Check("assets", spec.relative_path, PASS if ok else missing_status,
                                f"sha256 ok ({spec.purpose})" if ok else
                                f"sha256 {actual[:12]}… != pinned {spec.sha256[:12]}… ({spec.purpose})"))
    return checks


def check_easyocr_originals(root: Path, engine: str) -> List[Check]:
    # These upstream files are presence-checked only: the code does not pin them.
    status_if_missing = FAIL if engine == "easyocr" else WARN
    return [Check("assets", rel, PASS if (root / rel).is_file() else status_if_missing,
                  "present (unpinned upstream EasyOCR file)" if (root / rel).is_file()
                  else "missing (needed only for --engine easyocr)")
            for rel in EASYOCR_ORIGINAL_FILES]


def check_device(device: str, probe: Callable[[str], Optional[bool]]) -> Check:
    if device == "cpu":
        return Check("device", "detector", PASS, "cpu explicitly selected")
    available = probe(device)
    if available is None:
        return Check("device", device, FAIL, "torch could not be imported")
    if available:
        return Check("device", device, PASS, f"{device} available to this interpreter")
    hint = (" If this runs inside a sandboxed tool, rerun in a normal Terminal before changing device policy."
            if device == "mps" else "")
    return Check("device", device, FAIL, f"{device} unavailable; the launcher will return UNAVAILABLE.{hint}")


def torch_device_probe(device: str) -> Optional[bool]:
    try:
        import torch
    except Exception:
        return None
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if device == "cuda":
        return bool(torch.cuda.is_available())
    return True


def check_dataset(root: Path) -> List[Check]:
    """Read only the preparation metadata; never list or open split images."""
    meta_path = root / DATASET_DIR / "preparation_metadata.json"
    if not meta_path.is_file():
        return [Check("dataset", DATASET_DIR, WARN, "not present (not needed for the image demo)")]
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        return [Check("dataset", "preparation_metadata.json", WARN, f"unreadable: {exc}")]
    checks = []
    fingerprint = meta.get("dataset_fingerprint")
    checks.append(Check("dataset", "fingerprint", PASS if fingerprint == DATASET_FINGERPRINT else WARN,
                        f"{str(fingerprint)[:12]}…" + ("" if fingerprint == DATASET_FINGERPRINT
                                                       else f" != recorded {DATASET_FINGERPRINT[:12]}…")))
    counts = {k: v.get("images") for k, v in (meta.get("split_counts") or {}).items()}
    same = all(counts.get(k) == v for k, v in DATASET_SPLIT_IMAGES.items())
    checks.append(Check("dataset", "split image counts", PASS if same else WARN, json.dumps(counts, sort_keys=True)))
    verified = (meta.get("evidence_assurance") or {}).get("all_plate_texts_human_verified")
    checks.append(Check("dataset", "label provenance",
                        PASS if verified is False else WARN,
                        "publisher_mixed; not all labels human verified (expected)" if verified is False
                        else f"unexpected all_plate_texts_human_verified={verified!r}"))
    return checks


def port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def check_ports(ports: Mapping[int, str], probe: Callable[[int], bool] = port_in_use) -> List[Check]:
    return [Check("ports", f"{port} ({label})", WARN if probe(port) else PASS,
                  "already has a listener; check it before starting a duplicate" if probe(port) else "free")
            for port, label in ports.items()]


def check_frontend(root: Path) -> List[Check]:
    swc = root / NEXT_SWC_BINARY
    build = root / "apps/web/.next/BUILD_ID"
    return [
        Check("frontend", "next swc binary", PASS if swc.is_file() else WARN,
              "present" if swc.is_file() else "missing; `npm run build` will fail (restore 14.2.18 from lockfile, do not upgrade)"),
        Check("frontend", "production build", PASS if build.is_file() else WARN,
              "present (`npm run start` ready)" if build.is_file() else "not built; run `npm run build` in apps/web"),
    ]


# ── orchestration ────────────────────────────────────────────────────────────

def installed_version(name: str) -> Optional[str]:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_checks(engine: str, device: str, root: Path = ROOT, *, check_ports_enabled: bool = True) -> List[Check]:
    checks: List[Check] = []
    checks += check_interpreter(root, sys.prefix, sys.version_info[:3])
    pins_file = root / "requirements-anpr.txt"
    if pins_file.is_file():
        checks += check_packages(parse_pins(pins_file.read_text()), engine, installed_version)
    else:
        checks.append(Check("packages", "requirements-anpr.txt", WARN, "missing; versions not compared"))
    try:
        specs = pinned_assets(engine, root)
    except Exception as exc:  # report, do not crash: this is a diagnostic tool
        checks.append(Check("assets", "pin sources", FAIL, f"cannot read pins: {type(exc).__name__}: {exc}"))
    else:
        checks += check_assets(root, specs)
    checks += check_easyocr_originals(root, engine)
    checks.append(check_device(device, torch_device_probe))
    checks += check_dataset(root)
    if check_ports_enabled:
        checks += check_ports({8000: "API", 3000: "dashboard"})
    checks += check_frontend(root)
    return checks


def render(checks: Sequence[Check]) -> str:
    width = max((len(c.name) for c in checks), default=10)
    lines = [f"{c.status:<4}  {c.group:<9} {c.name:<{width}}  {c.detail}" for c in checks]
    totals = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL)}
    lines.append(f"\n{totals[PASS]} pass, {totals[WARN]} warn, {totals[FAIL]} fail")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only ANPR demo readiness check (no model loading).")
    parser.add_argument("--engine", choices=("rapidocr", "easyocr"), default="rapidocr")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    parser.add_argument("--skip-ports", action="store_true", help="do not probe local ports")
    args = parser.parse_args(argv)
    checks = run_checks(args.engine, args.device, check_ports_enabled=not args.skip_ports)
    if args.json:
        print(json.dumps({"engine": args.engine, "device": args.device,
                          "checks": [asdict(c) for c in checks]}, indent=2))
    else:
        print(f"RoadSense ANPR preflight — engine={args.engine} device={args.device}\n")
        print(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
