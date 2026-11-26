import argparse
import json
from pathlib import Path
import sys

import trace_digest


def frame_window(value):
    try:
        start, count = map(int, value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected START:COUNT") from None
    if start < 0 or count < 1:
        raise argparse.ArgumentTypeError("START must be nonnegative and COUNT positive")
    return start, count


def digest_capture(path, window):
    trace = trace_digest.load_trace(path)
    available = len(trace.presents) - 1
    start, count = window if window is not None else (0, available)
    if start < 0 or count < 1 or start + count > available:
        raise ValueError(f"{path}: cannot select {start}:{count} from {max(0, available)} complete frames")
    presents = trace.presents[start:start + count + 1]
    if any(left[0] >= right[0] for left, right in zip(presents, presents[1:])):
        raise ValueError(f"{path}: selected present timestamps must strictly increase")
    trace.presents = presents
    begin, end = presents[0][0], presents[-1][0]
    trace.slices = {tid: rows for tid, rows in trace.slices.items()
                    if any(ts < end and ts + dur > begin for ts, dur, *_ in rows)}
    trace.folded = {tid: rows for tid, rows in trace.folded.items() if any(begin <= ts < end for ts, _ in rows)}
    trace.batched = [row for row in trace.batched if row[1] < end and row[1] + row[2] > begin]
    bounds = {"start": start, "count": count, "start_ms": presents[0][0] / 1000,
              "end_ms": presents[-1][0] / 1000}
    return {"path": str(path), "window": bounds, "digest": trace_digest.digest(trace, worst=0, top=None)}


def main():
    parser = argparse.ArgumentParser(description="Compare independently selected windows of two APS5 JSON captures.")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    for arm in ("a", "b"):
        parser.add_argument(f"--{arm}-frames", type=frame_window, metavar="START:COUNT",
                            help="zero-based recorded present index and number of complete intervals")
    args = parser.parse_args()
    try:
        result = {"a": digest_capture(args.a, args.a_frames), "b": digest_capture(args.b, args.b_frames)}
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    for arm, capture in result.items():
        for loss in capture["digest"].get("event_loss", []):
            print(f"{arm.upper()}: {loss['thread']}: {loss['events_lost']} events lost at {loss['at_ms']:g} ms. "
                  "Timings describe retained events only.", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
