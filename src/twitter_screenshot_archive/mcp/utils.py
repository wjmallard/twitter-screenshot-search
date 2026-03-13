"""Shared MCP utilities."""


def _merge_similar_handles(
    user_counts: dict[str, int],
    primary: str | None = None,
) -> dict[str, int]:
    """Group handles by prefix overlap, keeping the longest variant.

    OCR frequently truncates handles — @bonzerba, @bon, @bonzerb are all
    fragments of @bonzerbarry.  This merges their counts into the longest
    matching handle.

    When *primary* is provided, also excludes any handle that is a prefix of
    or prefixed by the primary handle (catches both the full handle and its
    OCR fragments).
    """
    # Process longest first — the longest variant becomes canonical
    sorted_handles = sorted(user_counts, key=len, reverse=True)

    merged: dict[str, int] = {}

    for h in sorted_handles:
        # Skip fragments of the primary handle
        if primary and (h.startswith(primary) or primary.startswith(h)):
            continue

        # Check if h is a prefix of any existing canonical handle
        matched = False
        for canonical in merged:
            if canonical.startswith(h):
                merged[canonical] += user_counts[h]
                matched = True
                break

        if not matched:
            merged[h] = user_counts[h]

    return merged


def _dedup_handles(handles: list[str] | None) -> list[str]:
    """Deduplicate a handle list by prefix overlap, keeping the longest."""
    if not handles:
        return []

    result: list[str] = []
    for h in sorted(set(handles), key=len, reverse=True):
        if not any(existing.startswith(h) for existing in result):
            result.append(h)

    return result
