"""Tweet timestamp extraction from OCR text."""

import re
from datetime import datetime, timezone, timedelta


def parse_tz_offset(offset_str: str) -> timezone:
    """Parse an offset like '-05:00' or '+09:00' into a timezone."""
    sign = 1 if offset_str[0] == "+" else -1
    h, m = offset_str[1:].split(":")
    return timezone(timedelta(hours=sign * int(h), minutes=sign * int(m)))


# Absolute tweet timestamps from detail views.
# Formats seen in OCR:
#   1:28 AM - 1/13/26 - 6.3K Views
#   7:44 PM - Apr 21, 2025 - 96K Views
#   11:38 PM - 2025-09-23 - 36K Views
#   10:30 AM - 10/18/23 from Earth - 9 Views
_MONTH_NAMES = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_ABSOLUTE_TIME_RE = re.compile(
    r"(\d{1,2}:\d{2}\s*[AP]M)\s*"       # time: 1:28 AM
    r"[-–—]\s*"                           # separator
    r"("
    r"\d{1,2}/\d{1,2}/\d{2,4}"           # M/D/YY or M/D/YYYY
    r"|(?:" + _MONTH_NAMES + r")\s+\d{1,2},?\s*\d{4}"  # Apr 21, 2025
    r"|\d{4}-\d{2}-\d{2}"                # 2025-09-23
    r")",
    re.IGNORECASE,
)

_DATE_FORMATS = [
    "%m/%d/%y",      # 1/13/26
    "%m/%d/%Y",      # 1/13/2026
    "%b %d, %Y",     # Apr 21, 2025
    "%b %d %Y",      # Apr 21 2025 (no comma)
    "%Y-%m-%d",      # 2025-09-23
]


# Relative timestamps from timeline views.
# Appears after usernames: @user - 3h, @user : 12h, @user -1d
# Units: h (hours), m (minutes), d (days)
_RELATIVE_TIME_RE = re.compile(
    r"[-–—:]\s*(\d{1,3})([dhm])\s",
)

_RELATIVE_UNITS = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def _extract_absolute(
    ocr_text: str,
    tz_offset: str | None,
) -> tuple[datetime | None, str | None]:
    """Extract the last absolute timestamp (focal tweet in detail view)."""
    matches = _ABSOLUTE_TIME_RE.findall(ocr_text)
    if not matches:
        return None, None

    time_str, date_str = matches[-1]
    time_str = time_str.replace("\u00a0", " ").strip()
    date_str = date_str.replace(",", ", ").strip()
    date_str = " ".join(date_str.split())

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(f"{time_str} - {date_str}", f"%I:%M %p - {fmt}")
            if tz_offset:
                tz = parse_tz_offset(tz_offset)
                dt = dt.replace(tzinfo=tz).astimezone(timezone.utc)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt, "absolute"
        except ValueError:
            continue

    return None, None


def _extract_relative(
    ocr_text: str,
    capture_utc: datetime | None,
) -> tuple[datetime | None, str | None]:
    """Extract the first relative timestamp, anchored to capture time."""
    if capture_utc is None:
        return None, None

    match = _RELATIVE_TIME_RE.search(ocr_text)
    if not match:
        return None, None

    amount = int(match.group(1))
    unit = _RELATIVE_UNITS.get(match.group(2))
    if not unit:
        return None, None

    dt = capture_utc - timedelta(**{unit: amount})
    return dt, "relative"


def extract_tweet_time(
    ocr_text: str,
    capture_utc: datetime | None,
    tz_offset: str | None,
) -> tuple[datetime | None, str | None]:
    """Extract the focal tweet's timestamp from OCR text.

    Returns (tweet_time as UTC, source) where source is 'absolute',
    'relative', or None. Prefers absolute timestamps when available.
    """
    dt, source = _extract_absolute(ocr_text, tz_offset)
    if dt:
        return dt, source

    return _extract_relative(ocr_text, capture_utc)
