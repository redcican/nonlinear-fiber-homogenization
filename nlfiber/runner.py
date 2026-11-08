"""File-lock work queue sharded over GPUs (M0 of the plan).

A queue is a directory of JSON task files `<name>.json`. A worker claims a task by creating `<name>.lock`
atomically (O_EXCL), runs it, writes `<name>.done.json` (or `<name>.failed.json` with the traceback) and
removes the lock. Workers take tasks whose index has the parity of their shard (index % n_shards == shard),
so two cards split the queue by index parity; several workers per card take different tasks.

    python -m nlfiber.runner <queue dir> --shard 0 --n-shards 2 --gpu 0   # on card 0
    python -m nlfiber.runner <queue dir> --shard 1 --n-shards 2 --gpu 1   # on card 1

A task file holds {"module": "pkg.mod", "function": "run", "args": {...}}; the function receives the args dict
and returns a JSON-serializable dict. Stale locks older than `--stale` seconds are reclaimed (crashed workers).
"""
import argparse
import importlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def tasks(queue):
    return sorted(p for p in Path(queue).glob("*.json") if not p.name.endswith((".done.json", ".failed.json")))


def claim(path, stale):
    lock = path.with_suffix(".lock")
    if path.with_suffix(".done.json").exists():
        return False
    try:
        if lock.exists() and time.time() - lock.stat().st_mtime > stale:
            lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), time=time.time())))
    return True


def run_task(path):
    spec = json.loads(path.read_text())
    mod = importlib.import_module(spec["module"])
    fn = getattr(mod, spec.get("function", "run"))
    t0 = time.time()
    out = fn(spec.get("args", {}))
    out = dict(out or {})
    out["_seconds"] = time.time() - t0
    return out


def work(queue, shard=0, n_shards=1, stale=6 * 3600, once=False, heartbeat=300):
    done = 0
    while True:
        progressed = False
        for i, p in enumerate(tasks(queue)):
            if i % n_shards != shard or not claim(p, stale):
                continue
            progressed = True
            lock = p.with_suffix(".lock")
            try:
                out = run_task(p)
                p.with_suffix(".done.json").write_text(json.dumps(out, default=float))
                done += 1
            except Exception:  # noqa: BLE001  record the failure and go on
                p.with_suffix(".failed.json").write_text(json.dumps(dict(error=traceback.format_exc())))
            finally:
                if lock.exists():
                    lock.unlink()
        if once or not progressed:
            break
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--stale", type=float, default=6 * 3600)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    n = work(a.queue, a.shard, a.n_shards, a.stale, a.once)
    print(f"shard {a.shard}/{a.n_shards}: {n} tasks done")


if __name__ == "__main__":
    sys.exit(main())
