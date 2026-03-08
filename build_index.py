"""
One-time offline index builder.

Usage:
    python build_index.py [--data hafs_smart_v8.json] [--out quran_index.pkl]

The server auto-builds the index on first startup, so you only need this
script if you want to pre-build the index separately (e.g. in a Docker build
step) or to inspect index statistics.
"""

import argparse
import time

from quran_tracker.index import QuranIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Quran search index")
    parser.add_argument("--data", default="hafs_smart_v8.json")
    parser.add_argument("--out",  default="quran_index.pkl")
    parser.add_argument("--stats", action="store_true", help="Print frequency stats")
    args = parser.parse_args()

    t0 = time.perf_counter()
    idx = QuranIndex()
    idx.build_from_json(args.data)
    idx.save(args.out)
    elapsed = time.perf_counter() - t0

    print(f"Saved to {args.out}  ({elapsed:.1f}s)")

    if args.stats:
        _print_stats(idx)


def _print_stats(idx: QuranIndex) -> None:
    from collections import Counter
    freq = Counter(idx.words)
    print("\nTop 20 most frequent words:")
    for word, count in freq.most_common(20):
        print(f"  {word:<20} {count:5d}")

    print(f"\nRarest words (appearing once): {sum(1 for c in freq.values() if c == 1)}")
    print(f"Trigrams total  : {sum(len(v) for v in idx.trigram_idx.values())}")
    print(f"Unique trigrams : {len(idx.trigram_idx)}")

    # How many trigrams are unique (appear exactly once)?
    unique_tg = sum(1 for v in idx.trigram_idx.values() if len(v) == 1)
    print(f"Unique trigrams (appear once): {unique_tg}  "
          f"({100*unique_tg/max(len(idx.trigram_idx),1):.1f}%)")


if __name__ == "__main__":
    main()
