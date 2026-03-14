"""Standalone VLM description backfill entry point (tsa-describe)."""

from .vision import backfill_descriptions, load_vlm


def main():
    load_vlm()
    backfill_descriptions()
