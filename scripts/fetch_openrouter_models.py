#!/usr/bin/env python3
"""Refresh and inspect Ness Agent's runtime OpenRouter model catalog."""

from __future__ import annotations

import argparse

from ness_cli.model_catalog import catalog_cache_path, fetch_catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch OpenRouter and models.dev metadata and replace the cache",
    )
    args = parser.parse_args()
    if not args.refresh:
        parser.error("--refresh is required (runtime refresh is normally lazy)")
    records = fetch_catalog()
    print(f"cached {len(records)} models at {catalog_cache_path()}")


if __name__ == "__main__":
    main()
