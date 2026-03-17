"""
tests/run_all.py
================
Entry point — runs smoke test then full regression suite.

    python -m matching_engine.tests.run_all
"""

from matching_engine.config.match_config import MatchConfig
from matching_engine.tests import smoke_test, test_suite


def main() -> None:
    cfg = MatchConfig()

    smoke_test.run(cfg)

    print(f"\n{'═' * 70}")
    print("FULL TEST SUITE")
    print(f"{'═' * 70}")

    passed, failed = test_suite.run(cfg)
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()