"""Command line: python -m askdb.cli "How many patients are in Georgia?" """
from __future__ import annotations

import json
import sys

from .app import get_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m askdb.cli "your question"')
        raise SystemExit(1)
    result = get_pipeline().ask(" ".join(sys.argv[1:]))
    print(result["sql"], end="\n\n")
    if "result" in result:
        r = result["result"]
        print(" | ".join(r["columns"]))
        for row in r["rows"][:20]:
            print(" | ".join(str(v) for v in row))
    else:
        print(json.dumps({"error": result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
