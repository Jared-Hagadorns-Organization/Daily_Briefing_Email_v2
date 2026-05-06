"""Run the daily briefing graph."""
from __future__ import annotations
import sys

from src.graph import build_graph


def main() -> int:
    graph = build_graph()
    final_state = graph.invoke({})

    errors = final_state.get("errors") or []
    if errors:
        print("=== Errors during run ===", file=sys.stderr)
        for e in errors:
            print(f"  · {e}", file=sys.stderr)

    # Non-zero exit only if the email itself didn't go out
    body = final_state.get("email_body_html")
    if not body:
        return 1

    send_failed = any("SMTP send failed" in e for e in errors)
    return 1 if send_failed else 0


if __name__ == "__main__":
    sys.exit(main())
