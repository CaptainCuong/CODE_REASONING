#!/usr/bin/env python3
"""Stream-convert a top-level JSON array file to JSONL (one object per line).

Avoids loading the whole (multi-GB) input file into memory: it incrementally
decodes one array element at a time from a bounded read buffer.

Usage:
    python3 data_conversion/json_to_jsonl.py input.json output.jsonl
"""

import argparse
import json

CHUNK_SIZE = 1 << 20  # 1 MiB


def convert(in_path: str, out_path: str) -> int:
    decoder = json.JSONDecoder()
    count = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:

        def fill(buf: str) -> str:
            more = fin.read(CHUNK_SIZE)
            if not more:
                raise EOFError("unexpected end of file while parsing JSON array")
            return buf + more

        buf = fin.read(CHUNK_SIZE)
        idx = 0

        def skip_ws(buf: str, idx: int) -> tuple[str, int]:
            while True:
                while idx < len(buf) and buf[idx] in " \t\r\n":
                    idx += 1
                if idx == len(buf):
                    more = fin.read(CHUNK_SIZE)
                    if not more:
                        return buf, idx
                    buf += more
                    continue
                return buf, idx

        buf, idx = skip_ws(buf, idx)
        if buf[idx] != "[":
            raise ValueError(f"expected top-level JSON array, got {buf[idx]!r}")
        idx += 1

        while True:
            buf, idx = skip_ws(buf, idx)
            if buf[idx] == "]":
                break

            while True:
                try:
                    obj, end = decoder.raw_decode(buf, idx)
                    break
                except ValueError:
                    buf = fill(buf)

            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")
            count += 1
            idx = end
            buf = buf[idx:]
            idx = 0

            buf, idx = skip_ws(buf, idx)
            if buf[idx] == ",":
                idx += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="input JSON file (top-level array)")
    parser.add_argument("output", help="output JSONL file")
    args = parser.parse_args()

    n = convert(args.input, args.output)
    print(f"wrote {n:,} lines to {args.output}")


if __name__ == "__main__":
    main()
