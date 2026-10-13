"""
Build and validate the perturbation pool: 30 entries, comprising the 25 Unicode
`White_Space` code points and 5 zero-width characters.

Categories:
  - regular:       9 characters (HT, LF, VT, FF, CR, SPACE, NEL, LSEP, PSEP)
  - non_breaking: 16 characters (NBSP, OGHAM, EN/EM quad, EN/EM spaces, thin spaces, etc.)
  - zero_width:    5 characters (ZWSP, ZWNBSP, ZWNJ, ZWJ, Word Joiner)

Exports pool as JSON to results/whitespace_pool.json.
"""

import json
import logging
import os
import unicodedata

from config import WHITESPACE_POOL, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_char(char_entry: dict) -> bool:
    """
    Validate that a character entry meets whitespace criteria.
    Checks: character exists, has expected Unicode properties.
    """
    ch = char_entry["char"]
    code = char_entry["code"]

    # Check that the character is a single character
    if len(ch) != 1:
        logger.error(f"  FAIL: {code} is not a single character")
        return False

    # Check Unicode category
    cat = unicodedata.category(ch)
    name = unicodedata.name(ch, "UNKNOWN")

    # Whitespace categories: Zs (space separator), Zl (line separator),
    # Zp (paragraph separator), Cc (control, for tab/CR/LF)
    # Also Cf (format, for zero-width chars)
    valid_cats = {"Zs", "Zl", "Zp", "Cc", "Cf"}
    if cat not in valid_cats:
        logger.warning(f"  WARN: {code} category={cat}, name={name} — not standard whitespace category")

    logger.debug(f"  OK: {code} | category={cat} | name={name}")
    return True


def build_pool() -> dict:
    """
    Validate all 30 pool entries and return the pool metadata.
    """
    pool_data = {
        "total_characters": 0,
        "categories": {},
    }

    for category, chars in WHITESPACE_POOL.items():
        logger.info(f"Validating category: {category} ({len(chars)} chars)")
        valid_chars = []
        for entry in chars:
            if validate_char(entry):
                valid_chars.append(entry)

        pool_data["categories"][category] = {
            "count": len(valid_chars),
            "characters": valid_chars,
        }
        pool_data["total_characters"] += len(valid_chars)
        logger.info(f"  {len(valid_chars)}/{len(chars)} valid")

    return pool_data


def export_pool(pool_data: dict, output_path: str = None) -> str:
    """
    Export the whitespace pool to JSON.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "whitespace_pool.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pool_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Whitespace pool exported to {output_path}")
    return output_path


def print_summary(pool_data: dict):
    """Print a human-readable summary of the pool."""
    print()
    print("=" * 60)
    print(f"  Whitespace Pool: {pool_data['total_characters']} Unicode characters")
    print("=" * 60)
    for category, info in pool_data["categories"].items():
        print(f"\n  [{category}] — {info['count']} characters")
        for entry in info["characters"]:
            print(f"    {entry['code']:8s}  {entry['name']}")


# ─── CLI ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and validate the 30-code-point perturbation pool (25 White_Space + 5 zero-width)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/whitespace_pool.json)",
    )
    args = parser.parse_args()

    print("Building whitespace pool...")
    pool = build_pool()
    print_summary(pool)

    output = export_pool(pool, args.output)
    print(f"\nPool exported to: {output}")
