#!/usr/bin/env python
"""Check the published adapters against the course's Hugging Face rules.

The course asks for three things: every adapter on the Hugging Face Hub, the
repo public, and code that downloads every model and adapter by itself with no
manual step. This script checks them as a TA would see them, i.e. without your
login: the stored token is ignored throughout, so a private or gated repo fails
here exactly as it would for them.

  1. config    hf_adapters.json names a real repo
  2. public    the repo can be read anonymously
  3. complete  every local adapter folder is on the Hub, and its
               adapter_config.json and adapter_model.safetensors are
               byte-identical (sha256) to the files the reported results
               were produced with
  4. e2e       (--e2e) in a fresh copy of the code with no adapters/ folder and
               an empty Hugging Face cache, the handout's own commands run and
               fetch the base models and adapters themselves

    python src/verify_hf_adapters.py                  # checks 1-3, no GPU, ~1 min
    python src/verify_hf_adapters.py --e2e --gpu 5    # + check 4, ~10-20 min
    python src/verify_hf_adapters.py --e2e --gpu 5 \\
        --clone https://github.com/<user>/<repo>.git   # e2e from the pushed repo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
NEEDED_BY_CODE = ("best_sanitizer", "victim_raw", "victim_san")
BASE_MODELS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B")


class Report:
    def __init__(self):
        self.failed = False

    def ok(self, what: str, msg: str = "") -> None:
        print(f"  PASS  {what}" + (f" -- {msg}" if msg else ""), flush=True)

    def fail(self, what: str, msg: str) -> None:
        self.failed = True
        print(f"  FAIL  {what} -- {msg}", flush=True)

    def info(self, msg: str) -> None:
        print(f"        {msg}", flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_adapters(adir: str) -> List[str]:
    if not os.path.isdir(adir):
        return []
    return sorted(
        d for d in os.listdir(adir)
        if os.path.isfile(os.path.join(adir, d, "adapter_config.json"))
    )


def lfs_sha(sibling) -> Optional[str]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def anonymous_env(extra: Optional[dict] = None) -> dict:
    env = dict(os.environ)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env.update(extra or {})
    return env


# ----------------------------------------------------------------------------
# checks 1-3
# ----------------------------------------------------------------------------


def check_hub(rep: Report) -> Optional[dict]:
    from common import HF_ADAPTERS_FILE, hub_adapter_config

    print("== 1. config ==")
    cfg = hub_adapter_config()
    if not cfg:
        rep.fail("hf_adapters.json",
                 f"no usable repo_id in {HF_ADAPTERS_FILE}; set it to "
                 f"'<your-hf-username>/col873-a1-adapters' after uploading")
        return None
    repo, rev = cfg["repo_id"], (cfg.get("revision") or None)
    rep.ok("hf_adapters.json", f"repo_id={repo} revision={rev or 'main'}")

    print("== 2. public (anonymous access) ==")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.model_info(repo, revision=rev, files_metadata=True, token=False)
    except GatedRepoError:
        rep.fail("anonymous read", f"{repo} is gated; TAs would need to request access. "
                                   f"Turn gating off in the repo settings.")
        return None
    except RepositoryNotFoundError:
        rep.fail("anonymous read", f"{repo} is not readable without a login: it is private, "
                                   f"misnamed, or not uploaded yet. Make it public.")
        return None
    except Exception as exc:
        rep.fail("anonymous read", f"{type(exc).__name__}: {exc}")
        return None
    if getattr(info, "private", False):
        rep.fail("visibility", f"{repo} is private")
    else:
        rep.ok("anonymous read", f"{repo} is public (commit {info.sha[:12]})")
    if getattr(info, "gated", False):
        rep.fail("gating", f"{repo} is gated ({info.gated}); TAs cannot download it unattended")

    print("== 3. complete and identical to the local adapters ==")
    siblings = {s.rfilename: s for s in (info.siblings or [])}
    names = local_adapters(os.path.join(REPO, "adapters"))
    if not names:
        rep.info("no local adapters/ folder to compare against; checking presence only")
        names = list(NEEDED_BY_CODE)
    for name in sorted(set(names) | set(NEEDED_BY_CODE)):
        problems = []
        for f in REQUIRED_FILES:
            key = f"{name}/{f}"
            if key not in siblings:
                problems.append(f"{f} missing on the Hub")
                continue
            local = os.path.join(REPO, "adapters", name, f)
            if not os.path.isfile(local):
                continue
            remote = lfs_sha(siblings[key])
            if remote is None:  # small, non-LFS file: fetch it and hash
                from huggingface_hub import hf_hub_download
                with tempfile.TemporaryDirectory() as td:
                    p = hf_hub_download(repo, key, revision=rev, cache_dir=td, token=False)
                    remote = sha256_file(p)
            if remote != sha256_file(local):
                problems.append(f"{f} differs from adapters/{name}/{f}")
        if problems:
            rep.fail(f"adapter {name}", "; ".join(problems))
        else:
            size = getattr(siblings.get(f"{name}/adapter_model.safetensors"), "size", None)
            rep.ok(f"adapter {name}", f"{size / 2**20:.0f} MiB, sha256 matches" if size
                   else "present, sha256 matches")
    extra = sorted({k.split("/")[0] for k in siblings if "/" in k} - set(names))
    if extra:
        rep.info(f"also on the Hub (not in local adapters/): {', '.join(extra)}")
    if not rev:
        rep.info(f"tip: pin this exact upload by setting \"revision\": \"{info.sha}\" in "
                 f"hf_adapters.json, so later pushes to the repo cannot change what is graded")
    return cfg


# ----------------------------------------------------------------------------
# check 4
# ----------------------------------------------------------------------------


def check_e2e(rep: Report, cfg: dict, args) -> None:
    print("== 4. end-to-end from a fresh copy with an empty Hugging Face cache ==")
    parent = args.workdir_parent or os.path.dirname(REPO)
    wd = tempfile.mkdtemp(prefix="col873_hfcheck_", dir=parent)
    code, out, hf_home = (os.path.join(wd, d) for d in ("code", "out", "hf_home"))
    os.makedirs(out)
    os.makedirs(os.path.join(hf_home, "hub"))
    rep.info(f"workdir {wd}")
    try:
        if args.clone:
            r = subprocess.run(["git", "clone", "--depth", "1", args.clone, code],
                               capture_output=True, text=True)
            if r.returncode:
                rep.fail("git clone", r.stderr.strip()[-400:])
                return
            rep.ok("git clone", args.clone)
        else:
            shutil.copytree(REPO, code, ignore=shutil.ignore_patterns(
                "data", "adapters", "logs", "attack_records", "__pycache__", "*.zip",
                "submission", ".git"))
            rep.info("code copied from the working tree minus git-ignored paths "
                     "(pass --clone <url> to test the pushed repo instead)")

        if local_adapters(os.path.join(code, "adapters")):
            rep.fail("no adapters in git", "the clone contains adapters/; they belong on the "
                                           "Hub, not in the repository (moving them aside)")
            shutil.move(os.path.join(code, "adapters"), os.path.join(wd, "adapters_from_git"))
        if not os.path.isfile(os.path.join(code, "hf_adapters.json")):
            rep.fail("hf_adapters.json in repo", "missing from the clone -- commit it")
            return
        with open(os.path.join(code, "hf_adapters.json"), encoding="utf-8") as fh:
            if json.load(fh).get("repo_id") != cfg["repo_id"]:
                rep.fail("hf_adapters.json in repo",
                         "the committed copy names a different repo than the local one -- push it")

        # the dataset bundle comes from the course, not from Hugging Face
        data_dir = os.path.abspath(args.data_dir or os.path.join(REPO, "data"))
        if not os.path.isfile(os.path.join(data_dir, "partB_test.jsonl")):
            rep.fail("course data", f"no dataset bundle at {data_dir}; pass --data_dir "
                                    f"/path/to/the/provided/data")
            return
        os.symlink(data_dir, os.path.join(code, "data"))

        if args.reuse_base_models:
            from huggingface_hub import constants
            for m in BASE_MODELS:
                d = "models--" + m.replace("/", "--")
                src = os.path.join(constants.HF_HUB_CACHE, d)
                if os.path.isdir(src):
                    os.symlink(src, os.path.join(hf_home, "hub", d))
            rep.info("base models reused from the local cache (--reuse_base_models); "
                     "only the adapters are downloaded")
        else:
            rep.info("empty cache: base models (~4 GB) and adapters are all downloaded")

        env = anonymous_env({"HF_HOME": hf_home, "CUDA_VISIBLE_DEVICES": args.gpu,
                             "PYTHONUNBUFFERED": "1"})
        py = sys.executable
        partc = ["--ref_model", BASE_MODELS[1], "--corpus", "data/partC_corpus_raw.jsonl",
                 "--heldout", "data/partC_heldout.jsonl", "--canaries", "data/partC_canaries.jsonl",
                 "--pii_index", "data/partC_pii_index.jsonl",
                 "--targeted_file", "data/partC_targeted.jsonl",
                 "--attack", "targeted", "--smoke", "--seed", "42",
                 "--records_dir", os.path.join(out, "records")]
        runs = [
            ("sanitizer", "best_sanitizer",
             [py, "src/sanitize_eval.py", "--base_model", BASE_MODELS[0],
              "--adapter_path", "adapters/best_sanitizer", "--test_file", "data/partB_test.jsonl",
              "--system", "finetuned", "--limit", "20", "--seed", "42",
              "--output_predictions", os.path.join(out, "pred.jsonl"),
              "--report_file", os.path.join(out, "sanitizer.json")]),
            ("V-raw", "victim_raw",
             [py, "src/attack_extract.py", "--victim_model", BASE_MODELS[1],
              "--adapter_path", "adapters/victim_raw", *partc,
              "--report_file", os.path.join(out, "victim_raw.json")]),
            ("V-san", "victim_san",
             [py, "src/attack_extract.py", "--victim_model", BASE_MODELS[1],
              "--adapter_path", "adapters/victim_san", *partc,
              "--report_file", os.path.join(out, "victim_san.json")]),
        ]
        for label, name, cmd in runs:
            log = os.path.join(out, f"{name}.log")
            t0 = time.time()
            with open(log, "w") as fh:
                r = subprocess.run(cmd, cwd=code, env=env, stdout=fh, stderr=subprocess.STDOUT)
            text = open(log, errors="replace").read()
            if r.returncode:
                rep.fail(f"{label} run", f"exit {r.returncode}; last lines of {log}:\n"
                         + "\n".join("          " + l for l in text.strip().splitlines()[-8:]))
                continue
            if f"subfolder {name}) from the Hugging Face Hub" not in text:
                rep.fail(f"{label} run", "finished, but did not load the adapter from the Hub")
                continue
            rep.ok(f"{label} run", f"adapter fetched from the Hub, {time.time() - t0:.0f}s")

        cached = sorted(d for d in os.listdir(os.path.join(hf_home, "hub")) if d.startswith("models--"))
        want = ["models--" + cfg["repo_id"].replace("/", "--")]
        if not args.reuse_base_models:
            want += ["models--" + m.replace("/", "--") for m in BASE_MODELS]
        missing = [w for w in want if w not in cached]
        if missing:
            rep.fail("downloaded into the empty cache", f"missing {missing}")
        else:
            rep.ok("downloaded into the empty cache", ", ".join(w[8:].replace("--", "/") for w in want))

        # behaviour sanity check: the downloaded adapters should act like the reported ones
        try:
            s = json.load(open(os.path.join(out, "sanitizer.json")))
            rep.info(f"sanitizer on 20 test rows: exact micro-F1 {s['exact']['micro_f1']:.3f}, "
                     f"leak rate {s['leak_rate']:.3f}")
            leaks = {}
            for name in ("victim_raw", "victim_san"):
                t = json.load(open(os.path.join(out, f"{name}.json")))["targeted"]
                leaks[name] = (t.get("member") or {}).get("greedy_leak_rate")
            rep.info(f"targeted member greedy leak (smoke size): V-raw {leaks['victim_raw']:.3f} "
                     f"vs V-san {leaks['victim_san']:.3f} -- V-raw should leak clearly more")
        except Exception as exc:
            rep.info(f"(could not summarise outputs: {exc})")
    finally:
        if args.keep:
            rep.info(f"kept {wd}")
        else:
            shutil.rmtree(wd, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e2e", action="store_true", help="also run check 4 (needs a GPU)")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                   help="GPU for --e2e")
    p.add_argument("--clone", default="", help="git URL to clone for --e2e (default: copy "
                                               "the working tree minus git-ignored paths)")
    p.add_argument("--reuse_base_models", action="store_true",
                   help="--e2e: link the cached Qwen models instead of downloading ~4 GB")
    p.add_argument("--workdir_parent", default="", help="where --e2e builds its temp dir "
                                                        "(default: next to the repo)")
    p.add_argument("--data_dir", default="", help="--e2e: the provided dataset bundle "
                                                  "(default: data/ in the repo root)")
    p.add_argument("--keep", action="store_true", help="keep the --e2e temp dir")
    args = p.parse_args(argv)

    rep = Report()
    cfg = check_hub(rep)
    if args.e2e:
        if cfg is None:
            rep.fail("e2e", "skipped: checks 1-2 must pass first")
        else:
            check_e2e(rep, cfg, args)
    print()
    print("ALL CHECKS PASSED" if not rep.failed else "SOME CHECKS FAILED (see FAIL lines above)")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
