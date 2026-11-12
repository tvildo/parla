#!/usr/bin/env python3
"""Parla build tool (Python standard library only).

    python build.py wasm [--only I] [--list] [--out build/wasm]   engine wasm modules (Rust -> wasm32, wasm-opt)
    python build.py native                                      vi_cli (native engine runner)
    python build.py regress [--record]                          every model package, bit-exact vs tests/golden.json
    python build.py ts                                          the TypeScript for browsers -> build/js/ (tsc)
    python build.py check                                       type-check everything (TypeScript; Python with mypy via uv)
    python build.py extension [--zip]                           the browser extension -> build/extension/
    python build.py safari [--version X.Y.Z] [--ios]            the Safari app (macOS + Xcode) -> build/safari/
    python build.py sdk                                         the SDK as files for an app -> build/sdk/
    python build.py npm                                         the npm package @tvildo/parla -> build/npm/ + .tgz
    python build.py examples                                    build examples/ (react-vite, vanilla) on the package
    python build.py site [--out build/site]                     the website (landing, privacy, live demo)
    python build.py serve [PORT] [--root DIR]                   static server on 127.0.0.1 (default build/site)
    python build.py test [--build] [-- playwright args]         browser tests in Docker (tests/Dockerfile)
    python build.py screenshots                                 docs/images + website screenshots from the builds
    python build.py telemetry-deploy [--sample S] [--daily-limit N]  the report receiver -> Cloudflare (t.parla.cc)

Needs: Rust (rustup target add wasm32-unknown-unknown), binaryen's wasm-opt (WASM_OPT or on PATH), Node 24+ (it runs
the build scripts' TypeScript as is; `npm ci` here installs the compiler); Docker for `test`; uv for `check` (the
Python tools' locked environment, pyproject.toml + uv.lock). This script itself needs only python3.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
ENGINE = ROOT / "src" / "engine"
MODELS = ROOT / "models"
EXE = ".exe" if os.name == "nt" else ""


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if r.returncode:
        raise SystemExit(f"error: {' '.join(map(str, cmd))} exited with {r.returncode}")


def npm() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- wasm ------------------------------------------------------------------------------------------

# (name, cargo feature args). `rs` is the bundle every host uses (graph + dfn + rnnoise); the others are
# per-plugin modules (a host that needs only one model kind loads less).
WASM_VARIANTS = [
    ("rs", []),
    ("graph", ["--no-default-features", "--features", "graph"]),
    ("rnnoise", ["--no-default-features", "--features", "rnnoise"]),
    ("dfn", ["--no-default-features", "--features", "dfn"]),
    ("core", ["--no-default-features"]),
]
SIMD_FLAGS = {
    "simd": "-C target-feature=+simd128",
    "relaxed": "-C target-feature=+simd128,+relaxed-simd",
    "nosimd": "",
}
# (output name, cargo features, simd, crate dir). Last: the telemetry module (opt-in, loaded only when a host
# turns telemetry on) -> vi_otel.wasm.
WASM_JOBS = [(f"vi_{n}_{s}", f, s, "vi_rs") for n, f in WASM_VARIANTS for s in SIMD_FLAGS] + [
    ("vi_otel", [], "nosimd", "vi_otel_wasm")]
WASM_OPT_FEATURES = ["--enable-simd", "--enable-relaxed-simd", "--enable-bulk-memory",
                     "--enable-nontrapping-float-to-int", "--enable-sign-ext", "--enable-mutable-globals"]


def wasm_opt_bin() -> str:
    for cand in (os.environ.get("WASM_OPT"), shutil.which("wasm-opt")):
        if cand and (shutil.which(cand) or Path(cand).exists()):
            return cand
    raise SystemExit("error: wasm-opt (binaryen) not found: put it on PATH or set WASM_OPT")


def rust_sysroot() -> str:
    return subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True).stdout.strip()


def cmd_wasm(args: list[str]) -> None:
    """Non-kernel code at opt-level z, vi_kernels at O3 (Cargo.toml), then wasm-opt -Oz. Each module has its
    own target dir (RUSTFLAGS differ) under $VI_CACHE or build/cargo/. `--only I` builds one (parallel builds)."""
    ap = argparse.ArgumentParser(prog="build.py wasm")
    ap.add_argument("--only", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(BUILD / "wasm"))
    a = ap.parse_args(args)
    if a.list:
        for i, (n, _, _, _) in enumerate(WASM_JOBS):
            print(i, f"{n}.wasm")
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("VI_CACHE", BUILD / "cargo"))
    wo = wasm_opt_bin()
    for name, feats, simd, crate in [WASM_JOBS[a.only]] if a.only is not None else WASM_JOBS:
        tdir = cache / f"target_w_{name}"
        # no local paths in the binary (reproducible builds on any machine): the engine's source, and the toolchain's
        # (std's panic locations point into the local rust-src checkout, e.g. under the user's home directory)
        rf = " ".join(x for x in [SIMD_FLAGS[simd], f"--remap-path-prefix={ENGINE}=/vi",
                                  f"--remap-path-prefix={rust_sysroot()}=/rustc"] if x)
        t0 = time.time()
        run(["cargo", "build", "--locked", "--release", "--lib", "--target", "wasm32-unknown-unknown", "--target-dir", str(tdir), *feats],
            cwd=ENGINE / crate, env={"RUSTFLAGS": rf, "CARGO_PROFILE_RELEASE_OPT_LEVEL": "z", "CARGO_PROFILE_RELEASE_LTO": "false"})
        dst = out / f"{name}.wasm"
        run([wo, "-Oz", *WASM_OPT_FEATURES, str(tdir / "wasm32-unknown-unknown" / "release" / f"{crate}.wasm"), "-o", str(dst)])
        print(f"{dst.name}: {dst.stat().st_size} B  ({time.time() - t0:.1f} s)", flush=True)


# ---- native + regression ---------------------------------------------------------------------------

def vi_cli_path() -> Path:
    base = Path(os.environ["VI_CACHE"]) / "target_native" if os.environ.get("VI_CACHE") else ENGINE / "target"
    return base / "release" / f"vi_cli{EXE}"


def cmd_native(args: list[str]) -> None:
    extra = ["--target-dir", str(vi_cli_path().parents[1])] if os.environ.get("VI_CACHE") else []
    run(["cargo", "build", "--locked", "--release", "-p", "vi_rs", "--bin", "vi_cli", *extra], cwd=ENGINE)


GOLDEN = ROOT / "tests" / "golden.json"
# model -> (input rate, vi_cli flags)
REGRESS = {"passthrough": (24000, ""), "spectral": (16000, "--frame"), "gtcrn": (16000, "--frame"), "dtln": (16000, "--frame"),
           "nsnet2": (16000, "--frame"), "rnnoise": (48000, "--frame"), "deepfilternet3": (48000, "--frame")}


def test_signal(rate: int, seconds: float = 6.0) -> bytes:
    """Deterministic input: a voiced-like harmonic tone with a syllable rhythm + LCG noise + hum (no recording
    of anyone; the goldens only need a fixed, speech-shaped input)."""
    n, s, out = int(rate * seconds), 12345, []
    for i in range(n):
        t = i / rate
        f0 = 140 + 30 * math.sin(2 * math.pi * 0.7 * t)
        env = max(0.0, math.sin(2 * math.pi * 3.1 * t)) ** 2
        v = sum(math.sin(2 * math.pi * f0 * k * t) / k for k in range(1, 12) if f0 * k < rate / 2) * env
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(0.12 * v + 0.05 * (s / 2**32 - 0.5) + 0.02 * math.sin(2 * math.pi * 120 * t))
    return struct.pack(f"<{n}f", *out)


def platform_key() -> str:
    return f"{platform.machine()}-{sys.platform}"


def cmd_regress(args: list[str]) -> None:
    """Every model package through vi_cli, output sha256 vs tests/golden.json. Bit-exact PER PLATFORM: std
    exp/sin/cos come from the platform's libm (MSVC CRT vs glibc differ by ~1 ulp). A new platform: check the
    outputs numerically first, then --record."""
    ap = argparse.ArgumentParser(prog="build.py regress")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--build", action="store_true", help="build vi_cli first")
    a = ap.parse_args(args)
    if a.build or not vi_cli_path().exists():
        cmd_native([])
    plat = platform_key()
    gold = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {"outputs": {}}
    out = BUILD / "regress"
    out.mkdir(parents=True, exist_ok=True)
    fails = 0
    for model, (rate, flags) in REGRESS.items():
        inp = out / f"input_{rate}.f32"
        if not inp.exists():
            inp.write_bytes(test_signal(rate))
        dst = out / f"{model}.f32"
        run([str(vi_cli_path()), "run", str(MODELS / model / "manifest.json"), str(inp), str(dst), *flags.split(), "--reps", "1"])
        got = sha256(dst)
        if a.record:
            gold["outputs"].setdefault(model, {})[plat] = got
            print(f"REC  {model} {got[:16]} ({plat})", flush=True)
            continue
        want = gold["outputs"].get(model, {}).get(plat)
        if want is None:
            raise SystemExit(f"error: no golden for {model} on {plat}; check numerically, then --record")
        fails += got != want
        print(f"{'OK  ' if got == want else 'FAIL'} {model} {got[:16]}{'' if got == want else '  want ' + want[:16]}", flush=True)
    if a.record:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(gold, indent=2) + "\n", newline="\n")
        print(f"recorded {plat} in {GOLDEN}")
    if fails:
        raise SystemExit(f"{fails} model(s) differ from the golden outputs ({plat})")


# ---- typescript, extension, sdk, site --------------------------------------------------------------

def tsc(*args: str) -> None:
    """the TypeScript compiler of package.json (installed on first use)"""
    if not (ROOT / "node_modules" / "typescript").exists():
        run([npm(), "ci", "--no-audit", "--no-fund"], cwd=ROOT)
    run(["node", str(ROOT / "node_modules" / "typescript" / "bin" / "tsc"), *args], cwd=ROOT)


def cmd_ts(args: list[str]) -> None:
    """The code that runs in browsers (engine runtime, extension, SDK, telemetry Worker) -> build/js/: tsc type-checks
    it and writes the .d.ts files (tsconfig.json), tools/ts_emit.ts the JavaScript (types erased, every line where it is
    in the source). The builds copy from there; Node runs the other .ts files as they are."""
    shutil.rmtree(BUILD / "js", ignore_errors=True)
    tsc("-p", "tsconfig.json")
    run(["node", "--disable-warning=ExperimentalWarning", str(ROOT / "tools" / "ts_emit.ts")])


def cmd_check(args: list[str]) -> None:
    """Type-check everything: the TypeScript (browser code, build scripts and tools, browser tests) and the Python
    (this script and tools/*.py, mypy --strict in the environment locked by uv.lock)."""
    for project in ("tsconfig.json", "tsconfig.node.json", "tests/tsconfig.json"):
        tsc("-p", project, "--noEmit", "--emitDeclarationOnly", "false")
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("error: uv not found (https://docs.astral.sh/uv/getting-started/installation/); "
                         "it runs mypy in the locked Python environment (pyproject.toml, uv.lock)")
    run([uv, "run", "--locked", "mypy"], cwd=ROOT)


def cmd_extension(args: list[str]) -> None:
    cmd_ts([])
    run(["node", str(ROOT / "src" / "extension" / "build.ts"), *args])


SAFARI = ROOT / "src" / "extension" / "safari"


def safari_check() -> None:
    """The Xcode project lists the extension's top-level files one by one (src/extension/safari): a file the
    extension gained but the project doesn't list would silently be missing from the app."""
    pbx = (SAFARI / "Parla.xcodeproj" / "project.pbxproj").read_text(encoding="utf-8")
    listed = set(re.findall(r'path = "\.\./\.\./\.\./\.\./build/extension-safari/([^"/]+)"', pbx))
    built = {p.name for p in (BUILD / "extension-safari").iterdir()}
    if listed != built:
        raise SystemExit("error: the Safari Xcode project and build/extension-safari differ: "
                         f"not in the project {sorted(built - listed)}, missing from the build {sorted(listed - built)}. "
                         "Add / remove them in Xcode (Shared (Extension) > Resources, both extension targets): "
                         "src/extension/safari/README.md")


def cmd_safari(args: list[str]) -> None:
    """build/extension-safari, then (on macOS with Xcode) the app that carries it: build/safari/. Signing: with
    PARLA_APPLE_TEAM (a team ID) Xcode signs with that team; without it the app is signed to run locally only
    (Safari: Develop > Allow Unsigned Extensions)."""
    version = args[args.index("--version") + 1] if "--version" in args else None
    cmd_extension(["--browser", "safari", *(["--version", version] if version else [])])
    safari_check()
    if platform.system() != "Darwin" or not shutil.which("xcodebuild"):
        print("safari: build/extension-safari is ready; the app itself is built on a Mac with Xcode (python build.py safari)")
        return
    version = version or json.loads((BUILD / "extension-safari" / "manifest.json").read_text(encoding="utf-8"))["version"]
    major, minor, patch = (int(x) for x in version.split("."))
    common = [f"MARKETING_VERSION={version}", f"CURRENT_PROJECT_VERSION={major * 10000 + minor * 100 + patch}"]
    team = os.environ.get("PARLA_APPLE_TEAM")
    common += ([f"DEVELOPMENT_TEAM={team}", "CODE_SIGN_STYLE=Automatic", "-allowProvisioningUpdates"] if team
               else ["CODE_SIGN_IDENTITY=-", "CODE_SIGN_STYLE=Manual", "DEVELOPMENT_TEAM="])
    targets = [("Parla (macOS)", "platform=macOS")] + ([("Parla (iOS)", "generic/platform=iOS Simulator")] if "--ios" in args else [])
    for scheme, dest in targets:
        run(["xcodebuild", "-project", str(SAFARI / "Parla.xcodeproj"), "-scheme", scheme, "-configuration", "Release",
             "-destination", dest, "-derivedDataPath", str(BUILD / "safari"), "-quiet", *common, "build"])
    print(f"safari -> {BUILD / 'safari' / 'Build' / 'Products'} (Parla {version})")


def cmd_sdk(args: list[str]) -> None:
    cmd_ts([])
    run(["node", str(ROOT / "src" / "sdk" / "build.ts"), *args])


def cmd_npm(args: list[str]) -> None:
    """build/npm/ (the package with its engine files) and build/tvildo-parla-<version>.tgz (npm pack)."""
    cmd_ts([])
    run(["node", str(ROOT / "src" / "sdk" / "npm.ts"), "--pack", *args])


def cmd_examples(args: list[str]) -> None:
    """examples/ on the packed package: install it fresh, build what has a build."""
    for ex in ("vanilla", "react-vite"):
        d = ROOT / "examples" / ex
        shutil.rmtree(d / "node_modules" / "@tvildo", ignore_errors=True)   # the same version number, new contents
        # the lockfile pins the checksum of the packed file, which changes with every SDK build: drop just that one
        lock = d / "package-lock.json"
        if lock.exists():
            data = json.loads(lock.read_text(encoding="utf-8"))
            entry = data.get("packages", {}).get("node_modules/@tvildo/parla")
            if entry and entry.pop("integrity", None):
                lock.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
        run([npm(), "ci", "--no-audit", "--no-fund"], cwd=d)   # strictly the lockfile (.npmrc: no install scripts)
        if (d / "vite.config.ts").exists():
            run([npm(), "run", "build"], cwd=d)




def relative_links(root: Path) -> None:
    """Astro writes root-absolute links (/_astro/..., /ka/demo/); make them relative to each page, so one build works
    at the root and under a prefix (the /d/ preview). Full URLs (https://, //) stay as they are."""
    attr = re.compile(r'(\s(?:href|src|component-url|renderer-url|before-hydration-url)=")/(?!/)')
    for f in root.rglob("*.html"):
        depth = len(f.parent.relative_to(root).parts)
        prefix = "../" * depth or "./"
        f.write_text(attr.sub(lambda m: m.group(1) + prefix, f.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")


def cmd_site(args: list[str]) -> None:
    """The website (src/site, Astro): landing, privacy and the live demo in every language of src/i18n. The demo uses
    the npm package (build/npm, `build.py npm` first): its engine files become hashed assets under _astro/."""
    ap = argparse.ArgumentParser(prog="build.py site")
    ap.add_argument("--out", default=str(BUILD / "site"))
    ap.add_argument("--wasm", default=str(BUILD / "wasm"))
    a = ap.parse_args(args)
    out = Path(a.out)
    run(["node", str(ROOT / "tools" / "i18n_check.ts")])
    if not (BUILD / "npm" / "package.json").exists():
        cmd_npm(["--wasm", a.wasm])
    site = ROOT / "src" / "site"
    if not (site / "node_modules").exists():
        run([npm(), "ci"], cwd=site)
    if out.exists():
        shutil.rmtree(out)
    # the install page links the release's files by name (parla-chrome-v<version>.zip): the version the npm package was
    # built with (CI builds it with the release version first)
    version = json.loads((BUILD / "npm" / "package.json").read_text(encoding="utf-8"))["version"]
    run([npm(), "run", "build"], cwd=site, env={"PARLA_VERSION": version})
    shutil.copytree(site / "dist", out, dirs_exist_ok=True)
    # the licences of what the site serves: Parla's NOTICE + the demo's model packages (the npm package's models)
    shipped = sorted(p.name for p in (BUILD / "npm" / "dist" / "models").iterdir() if p.is_dir())
    text = [(ROOT / "NOTICE").read_text(encoding="utf-8").rstrip()]
    for m in shipped:
        lic = ROOT / "models" / m / "LICENSE"
        if lic.exists():
            text.append(f"==== models/{m}/LICENSE ====\n\n" + lic.read_text(encoding="utf-8").rstrip())
    text.append("==== Parla: LICENSE ====\n\n" + (ROOT / "LICENSE").read_text(encoding="utf-8").rstrip())
    (out / "licenses.txt").write_text("\n\n".join(text) + "\n", encoding="utf-8", newline="\n")
    relative_links(out)
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"site: {out} ({n} files)")


# ---- serve -----------------------------------------------------------------------------------------

def cmd_serve(args: list[str]) -> None:
    """Static server on 127.0.0.1 (stdlib) with byte ranges (media seeking) and no caching."""
    ap = argparse.ArgumentParser(prog="build.py serve")
    ap.add_argument("port", nargs="?", type=int, default=8765)
    ap.add_argument("--root", default=str(BUILD / "site"))
    a = ap.parse_args(args)
    root = Path(a.root)

    class H(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript",
                          ".json": "application/json", ".bin": "application/octet-stream", ".webm": "video/webm"}
        _left: int | None = None   # bytes still to send of a range answer

        def __init__(self, *x: Any, **k: Any) -> None:
            super().__init__(*x, directory=str(root), **k)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def send_head(self) -> BinaryIO | None:
            # single byte ranges, so media elements can seek (the stdlib ignores Range)
            rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            p = Path(self.translate_path(self.path))
            if not rng or not p.is_file() or rng.group(1) + rng.group(2) == "":
                return super().send_head()
            size = p.stat().st_size
            lo, hi = rng.groups()
            start, end = (int(lo), min(int(hi) if hi else size - 1, size - 1)) if lo else (max(0, size - int(hi)), size - 1)
            if start >= size:
                self.send_error(416)
                return None
            f = open(p, "rb")
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(p)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self._left = end - start + 1
            return f

        def copyfile(self, src: Any, dst: Any) -> None:
            left = self._left
            if left is None:
                return super().copyfile(src, dst)
            while left > 0:
                b = src.read(min(left, 1 << 16))
                if not b:
                    break
                dst.write(b)
                left -= len(b)
            self._left = None

        def log_message(self, *x: Any) -> None:
            pass

    print(f"serving {root} at http://127.0.0.1:{a.port}/", flush=True)
    # no SO_REUSEADDR: on Windows it lets a second server bind the same port, and requests then go to
    # whichever one the OS picks
    http.server.ThreadingHTTPServer.allow_reuse_address = False
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def cmd_test(args: list[str]) -> None:
    """The browser tests in a container (tests/Dockerfile: Playwright's Chromium, Node, a Firefox release with
    PulseAudio): the extension in Chromium and in Firefox, the website, the SDK, on this tree's build outputs. `--build` builds them first (wasm, extension, site)."""
    if "--build" in args:
        args = [a for a in args if a != "--build"]
        cmd_wasm([])
        cmd_extension([])
        cmd_extension(["--browser", "firefox"])
        cmd_npm([])
        cmd_site([])
        cmd_examples([])
    for need in ("extension/manifest.json", "extension-firefox/manifest.json", "site/index.html", "wasm/vi_rs_simd.wasm", "npm/package.json"):
        if not (BUILD / need).exists():
            raise SystemExit(f"error: build/{need} missing (python build.py test --build)")
    run(["docker", "build", "-q", "-t", "parla-tests", str(ROOT / "tests")])
    pw = args[args.index("--") + 1:] if "--" in args else []
    cmd = ["sh", "run_all.sh", *pw]   # Chromium (Playwright), the SDK, then Firefox (tests/run_all.sh)
    # node_modules in a volume: the host's (Windows) copy stays out of the Linux container
    run(["docker", "run", "--rm", "--ipc=host", "-v", f"{ROOT}:/oss", "-v", "parla-tests-modules:/oss/tests/node_modules", "parla-tests", *cmd])


def cmd_screenshots(args: list[str]) -> None:
    """docs/images/*.png and src/site/src/assets/screenshots/ from the current builds (extension, site), taken on this
    machine's Chromium (its fonts). Rebuild the site afterwards to ship the new images."""
    run([npm(), "exec", "--", "playwright", "test", "--project", "screenshots", *args], cwd=ROOT / "tests", env={"SCREENSHOTS": "1"})

TELEMETRY_HOST = "t.parla.cc"
TELEMETRY_WORKER = "parla-telemetry"
CF_API = "https://api.cloudflare.com/client/v4"


def cmd_telemetry_deploy(args: list[str]) -> None:
    """The extension's report receiver (src/telemetry/worker.ts, as JavaScript) as a Cloudflare Worker at
    https://t.parla.cc/v1/e: Workers Logs, Analytics Engine dataset `parla_reports`, a Durable Object counting events
    per day. Idempotent: uploads the current worker.js and (re)attaches the domain. Credentials only from the
    environment: CLOUDFLARE_API_TOKEN (Workers Scripts: Edit, Workers Custom Domains: Edit, Zone: Read) and
    CLOUDFLARE_ACCOUNT_ID. Never printed.
        telemetry-deploy [--sample 0..1] [--daily-limit N] [--log-sampling 0..1] [--host H] [--name N]
    --sample: share of installs that report (sent back to every extension; 0 = reports off, no update needed);
    --daily-limit: events accepted per UTC day, then extensions pause until midnight; --log-sampling: share of log
    lines kept. A fork deploys its own receiver with --host / --name (and its extension's endpoint)."""
    import urllib.error
    import urllib.request
    import uuid
    ap = argparse.ArgumentParser(prog="build.py telemetry-deploy")
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--daily-limit", type=int, default=80000)
    ap.add_argument("--log-sampling", type=float, default=1.0)
    ap.add_argument("--host", default=TELEMETRY_HOST)
    ap.add_argument("--name", default=TELEMETRY_WORKER)
    a = ap.parse_args(args)
    if not (0 <= a.sample <= 1 and 0 < a.log_sampling <= 1 and a.daily_limit > 0):
        raise SystemExit("error: --sample 0..1, --log-sampling (0, 1], --daily-limit > 0")
    token, acct = os.environ.get("CLOUDFLARE_API_TOKEN", ""), os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not token or not acct:
        raise SystemExit("error: set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID")

    def cf(method: str, path: str, body: object = None, raw: bytes | None = None, ctype: str = "application/json") -> Any:
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(CF_API + path, method=method, data=data,
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            d = json.loads(e.read() or b"{}")
        if not d.get("success"):
            raise SystemExit(f"error: cloudflare {method} {path.split('?')[0]}: {[x.get('message') for x in d.get('errors', [])]}")
        return d["result"]

    cmd_ts([])
    code = (BUILD / "js" / "src" / "telemetry" / "worker.js").read_text(encoding="utf-8")
    scripts = cf("GET", f"/accounts/{acct}/workers/scripts")
    current = next((x for x in scripts if x.get("id") == a.name), None)
    meta: dict[str, Any] = {"main_module": "worker.js", "compatibility_date": "2026-10-01",
                  "bindings": [
                      {"type": "plain_text", "name": "SAMPLE", "text": str(a.sample)},
                      {"type": "plain_text", "name": "DAILY_LIMIT", "text": str(a.daily_limit)},
                      {"type": "analytics_engine", "name": "AE", "dataset": "parla_reports"},
                      {"type": "durable_object_namespace", "name": "BUDGET", "class_name": "Budget"},
                  ],
                  "observability": {"enabled": True, "head_sampling_rate": a.log_sampling}}
    if not (current and current.get("migration_tag")):   # the Durable Object class, created once (SQLite-backed)
        meta["migrations"] = {"new_tag": "v1", "new_sqlite_classes": ["Budget"]}
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n"
            f"{json.dumps(meta)}\r\n--{b}\r\nContent-Disposition: form-data; name=\"worker.js\"; filename=\"worker.js\"\r\n"
            f"Content-Type: application/javascript+module\r\n\r\n{code}\r\n--{b}--\r\n").encode()
    cf("PUT", f"/accounts/{acct}/workers/scripts/{a.name}", raw=body, ctype=f"multipart/form-data; boundary={b}")
    print(f"telemetry: worker {a.name} uploaded (sample {a.sample}, daily limit {a.daily_limit}, log sampling {a.log_sampling})")
    zones = cf("GET", "/zones?per_page=50")
    zone = max((z for z in zones if a.host.endswith("." + z["name"])), key=lambda z: len(z["name"]), default=None)
    if zone is None:
        raise SystemExit(f"error: no zone for {a.host} in this Cloudflare account")
    cf("PUT", f"/accounts/{acct}/workers/domains", {"hostname": a.host, "service": a.name, "zone_id": zone["id"], "environment": "production"})
    print(f"telemetry: https://{a.host}/v1/e -> worker {a.name}")



COMMANDS = {"wasm": cmd_wasm, "native": cmd_native, "regress": cmd_regress, "ts": cmd_ts, "check": cmd_check, "extension": cmd_extension, "safari": cmd_safari,
            "sdk": cmd_sdk, "npm": cmd_npm, "examples": cmd_examples, "screenshots": cmd_screenshots, "site": cmd_site, "serve": cmd_serve, "test": cmd_test,
            "telemetry-deploy": cmd_telemetry_deploy}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(2)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
