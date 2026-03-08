"""
Arabic text normalization for Quran STT matching.

Rules applied (to both corpus and STT output identically):
  1. Strip diacritics (harakat, shadda, sukun, etc.)
  2. Strip tatweel (kashida stretch character)
  3. Normalize all alef variants → bare alef (ا)
  4. Normalize hamza-on-waw (ؤ→و) and hamza-on-ya (ئ→ي)
  5. Normalize alef maqsura (ى→ي)
  6. Normalize ta marbuta (ة→ه)  — both sound the same in STT output

These rules are intentionally aggressive so that STT output and Quran text
converge to the same form despite different orthographic conventions.
"""

import re
import unicodedata
from typing import List

# ── Regex patterns ────────────────────────────────────────────────────────────

_DIACRITICS = re.compile(
    "["
    "\u064B-\u065F"  # Fathatan … Sukun  (standard tashkeel)
    "\u0610-\u061A"  # Extended Arabic A  (sallallahou, etc.)
    "\u06D6-\u06DC"  # Quranic annotation signs
    "\u06DF-\u06E4"  # More annotation signs
    "\u06E7-\u06ED"  # Yet more annotation
    "\u0670"         # Arabic letter superscript alef
    "]"
)

_TATWEEL = re.compile("\u0640")  # ـ  kashida / tatweel

# ── Translation tables ────────────────────────────────────────────────────────

# All alef variants → plain alef ا (U+0627)
_ALEF_TABLE = str.maketrans(
    "\u0623\u0625\u0622\u0671",   # أ إ آ ٱ
    "\u0627\u0627\u0627\u0627",   # → ا ا ا ا
)

# Hamza on carriers → base letter
_HAMZA_TABLE = str.maketrans(
    "\u0624\u0626",  # ؤ ئ
    "\u0648\u064A",  # → و ي
)

# Alef maqsura + ta marbuta → ya + ha
_TAIL_TABLE = str.maketrans(
    "\u0649\u0629",  # ى ة
    "\u064A\u0647",  # → ي ه
)


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_word(word: str) -> str:
    """Return the canonical normalized form of an Arabic word."""
    word = _DIACRITICS.sub("", word)
    word = _TATWEEL.sub("", word)
    word = word.translate(_ALEF_TABLE)
    word = word.translate(_HAMZA_TABLE)
    word = word.translate(_TAIL_TABLE)
    return word.strip()


def tokenize(text: str) -> List[str]:
    """Split Arabic text into normalized word tokens, dropping empty strings."""
    return [nw for nw in (normalize_word(t) for t in text.split()) if nw]
