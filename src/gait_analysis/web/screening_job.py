"""Subprocess entry point for a single-phone screening job.

Run in its OWN process by the web JobManager so the heavy MediaPipe pose pass
doesn't starve the uvicorn event loop (a CPU-bound thread inside the web process
makes the app unresponsive — even /health times out). The manager thread just
waits on this subprocess, which releases the GIL and keeps the web responsive.

Usage:  python -m gait_analysis.web.screening_job <video> <outdir> [subject]
Writes <outdir>/report.html on success; exits non-zero with a message on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print("usage: screening_job <video> <outdir> [subject] [task] [height_cm] [weight_kg]",
              file=sys.stderr)
        return 2
    video, outdir = argv[0], argv[1]
    subject = argv[2] if len(argv) > 2 else ""
    task = argv[3] if len(argv) > 3 else "gait"

    def _num(i):
        try:
            return float(argv[i]) if len(argv) > i and argv[i] else None
        except ValueError:
            return None

    from ..pipeline import run_screening
    result = run_screening(video, outdir, subject=subject, task=task,
                           height_cm=_num(4), weight_kg=_num(5))
    # Standardize the served filename (the web serves <sid>/report.html).
    Path(outdir, "report.html").write_text(Path(result["report"]).read_text())
    print("screening OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
