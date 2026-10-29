"""
Pure string parsers for the raw attribute values the structuring pass
(texthandling/structure.py) produces, e.g. "8,5 m", "max. 48°",
"25 - 35°", "WA II". No I/O, no OGR -- every function takes a string and
returns plain Python values, or raises ParseError so mapping.py can turn
the failure into a report entry instead of silently dropping the value.
"""

import re
from typing import NamedTuple


class ParseError(ValueError):
    pass


class Range(NamedTuple):
    """min/max bounds of a value; `exact` set when the plan gives one number
    with no "min."/"max." qualifier (e.g. "DN 45°")."""
    min: float | None = None
    max: float | None = None
    exact: float | None = None


_NUMBER = r"\d+(?:[.,]\d+)?"
_ROMAN = {"I": 1, "V": 5, "X": 10}


def to_float(raw):
    """'8,5 m' -> 8.5, '0,4' -> 0.4, '48°' -> 48.0"""
    match = re.search(_NUMBER, raw)
    if match is None:
        raise ParseError(f"no number in {raw!r}")
    return float(match.group().replace(",", "."))


def parse_range(raw):
    """
    '25 - 35°'   -> Range(min=25, max=35)
    'max. 48°'   -> Range(max=48)
    'min. 30°'   -> Range(min=30)
    '45°'        -> Range(exact=45)
    """
    numbers = [float(n.replace(",", ".")) for n in re.findall(_NUMBER, raw)]
    if len(numbers) == 2:
        low, high = sorted(numbers)
        return Range(min=low, max=high)
    if len(numbers) != 1:
        raise ParseError(f"expected one or two numbers in {raw!r}")
    lowered = raw.lower()
    if "max" in lowered:
        return Range(max=numbers[0])
    if "min" in lowered:
        return Range(min=numbers[0])
    return Range(exact=numbers[0])


def roman_to_int(raw):
    """'II' -> 2, 'IV' -> 4; also accepts arabic digits ('2' -> 2)."""
    raw = raw.strip().upper()
    if raw.isdigit():
        return int(raw)
    if not raw or any(c not in _ROMAN for c in raw):
        raise ParseError(f"not a roman numeral: {raw!r}")
    total = 0
    for current, following in zip(raw, raw[1:] + " "):
        value = _ROMAN[current]
        total += -value if _ROMAN.get(following, 0) > value else value
    return total


def split_nutzung_geschosse(raw):
    """
    Splits a combined use-type/storey label as found in the plan's
    Nutzungsschablone:
        'WA II'   -> ('WA', 2)
        'WA2'     -> ('WA', 2)
        'MI'      -> ('MI', None)
    Returns (use type abbreviation, number of full storeys or None).
    """
    match = re.fullmatch(r"\s*([A-Za-z]{2,3}?)\s*([IVX]+|\d+)?\s*", raw)
    if match is None:
        raise ParseError(f"unrecognized use type/storeys: {raw!r}")
    abbreviation, storeys = match.groups()
    return abbreviation.upper(), (roman_to_int(storeys) if storeys else None)
