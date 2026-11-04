"""Write <site>/plugins/index.json: wasm modules with size + sha256 (the browser store verifies).
    python3 tools/wasm_index.py [site_dir]      (default: build/site)"""
import hashlib
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "build" / "site"
files = {}
for p in sorted(list(d.glob("vi_rs_*.wasm")) + list((d / "plugins").glob("*.wasm"))):
    files[str(p.relative_to(d))] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
idx = {"plugins": {"passthrough": "core", "graph": "graph", "dfn": "dfn", "rnnoise": "rnnoise"},
       "bundle_kinds": ["passthrough", "graph", "dfn", "rnnoise"],
       "bundle": "vi_rs_{build}.wasm", "module": "plugins/vi_{plugin}_{build}.wasm", "files": files}
(d / "plugins" / "index.json").write_text(json.dumps(idx, indent=1))
print(f"{len(files)} wasm files indexed")
