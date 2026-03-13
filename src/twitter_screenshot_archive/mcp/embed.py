"""Standalone embedding backfill entry point (tsa-embed)."""

from .embedding import backfill_embeddings, load_model


def main():
    load_model()
    backfill_embeddings()
