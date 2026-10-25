#!/usr/bin/env python3
"""Run every verification script of the two-readings investigation and summarise the outcome.

    python3 investigations/two_readings/verify/run_all.py

Each script prints one 'PASS <id>: ...' line per verified claim and exits 0 only if all of its checks pass.
The derivation scripts are r1-r3; s1-s3 are the independent adversarial re-checks written by a second author.
A summary is written to investigations/two_readings/verify/RESULTS.md.  Exit status 0 iff every script passes.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("[rs][0-9]*.py"))


def main() -> int:
    rows, ok_all = [], True
    for script in SCRIPTS:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=HERE, timeout=1800)
        dt = time.time() - t0
        out = proc.stdout.splitlines()
        n_pass = sum(1 for line in out if line.startswith("PASS"))
        n_fail = sum(1 for line in out if line.startswith("FAIL"))
        ok = proc.returncode == 0 and n_fail == 0
        ok_all &= ok
        rows.append((script.name, ok, n_pass, n_fail, dt))
        print(f"{'OK  ' if ok else 'FAIL'} {script.name:40s} {n_pass:3d} PASS {n_fail:3d} FAIL  {dt:6.1f}s")
        if not ok:
            print("\n".join(out[-20:]))
            print(proc.stderr[-2000:])
    lines = ["# Verification results", "", "| script | status | PASS | FAIL | time (s) |", "|---|---|---|---|---|"]
    lines += [f"| `{n}` | {'ok' if ok else '**FAIL**'} | {p} | {f} | {t:.1f} |" for n, ok, p, f, t in rows]
    lines += ["", f"Total: {sum(r[2] for r in rows)} PASS lines, {sum(r[3] for r in rows)} FAIL lines, "
              f"{'all scripts pass' if ok_all else 'SOME SCRIPTS FAIL'}."]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
