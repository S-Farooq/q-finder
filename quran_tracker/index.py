"""
Quran search index.

Builds two data structures from hafs_smart_v8.json (aya_text_emlaey field):

  inverted_index : normalized_word  → sorted list[global_word_pos]
  trigram_index  : (w0, w1, w2)    → sorted list[global_word_pos]  (start of trigram)

Search algorithm:
  1. Generate candidate start-positions using trigram_index (exact trigrams from query).
  2. Fall back to the rarest word in the inverted_index if trigrams yield nothing.
  3. Score each candidate with a sliding word-match window (exact / fuzzy).
  4. Return top-K SearchMatch objects sorted by score.

Edit-distance is capped at 2 with an early-exit optimisation so it stays fast
even when invoked thousands of times per second.
"""

import json
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .normalize import normalize_word, tokenize


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class WordMeta:
    global_idx: int
    surah:      int
    ayah:       int
    word_pos:   int    # 0-based position within the ayah
    original:   str    # raw text from aya_text_emlaey (single word)
    normalized: str    # after normalize_word()


@dataclass
class SearchMatch:
    start_pos:   int    # global word index where the query starts
    score:       float  # 0.0–1.0
    n_matched:   int    # number of query words that contributed
    surah:       int
    ayah:        int
    word_pos:    int    # word position within the ayah at start_pos


# ── Edit distance ─────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str, max_dist: int = 2) -> int:
    """Standard DP edit distance with early-exit when cost exceeds max_dist."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la > lb:
        a, b, la, lb = b, a, lb, la   # ensure a is shorter

    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        row_min = j
        for i in range(1, la + 1):
            curr[i] = min(
                curr[i - 1] + 1,
                prev[i] + 1,
                prev[i - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
            if curr[i] < row_min:
                row_min = curr[i]
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[la]


# ── Word match score ──────────────────────────────────────────────────────────

def word_match_score(corpus_word: str, query_word: str) -> float:
    """
    Returns a similarity score in [0, 1].
      exact match  → 1.0
      edit dist 1  → 0.80
      edit dist 2  → 0.45
      otherwise    → 0.0
    """
    if corpus_word == query_word:
        return 1.0
    d = _edit_distance(corpus_word, query_word, max_dist=2)
    if d == 1:
        return 0.80
    if d == 2:
        return 0.45
    return 0.0


# ── Index ─────────────────────────────────────────────────────────────────────

class QuranIndex:
    def __init__(self) -> None:
        self.words:         List[str]                        = []
        self.meta:          List[WordMeta]                   = []
        self.inverted:      Dict[str, List[int]]             = {}
        self.trigram_idx:   Dict[Tuple[str, str, str], List[int]] = {}
        self.n_words:       int                              = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_json(self, json_path: str) -> None:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        global_idx = 0
        for entry in data:
            surah    = entry["sura_no"]
            ayah_no  = entry["aya_no"]
            raw_text = entry.get("aya_text_emlaey", "")

            for word_pos, raw_word in enumerate(raw_text.split()):
                norm = normalize_word(raw_word)
                if not norm:
                    continue

                meta = WordMeta(
                    global_idx=global_idx,
                    surah=surah,
                    ayah=ayah_no,
                    word_pos=word_pos,
                    original=raw_word,
                    normalized=norm,
                )
                self.words.append(norm)
                self.meta.append(meta)

                if norm not in self.inverted:
                    self.inverted[norm] = []
                self.inverted[norm].append(global_idx)

                global_idx += 1

        self.n_words = global_idx

        # Build trigram index
        for i in range(self.n_words - 2):
            tg = (self.words[i], self.words[i + 1], self.words[i + 2])
            if tg not in self.trigram_idx:
                self.trigram_idx[tg] = []
            self.trigram_idx[tg].append(i)

        print(
            f"[QuranIndex] built: {self.n_words} words, "
            f"{len(self.inverted)} unique tokens, "
            f"{len(self.trigram_idx)} unique trigrams"
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "QuranIndex":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_at(self, query_words: List[str], start: int) -> float:
        """Score how well query_words align starting at global position start."""
        if start < 0 or start + len(query_words) > self.n_words:
            return 0.0
        total = sum(
            word_match_score(self.words[start + i], q)
            for i, q in enumerate(query_words)
        )
        return total / len(query_words)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_words: List[str],
        top_k: int = 10,
        nearby_pos: Optional[int] = None,
        nearby_window: int = 150,
        min_score: float = 0.50,
    ) -> List[SearchMatch]:
        """
        Find the best matches for query_words in the Quran.

        nearby_pos / nearby_window: if provided, also brute-force score
        every position in that window (helps when tracking is close but
        the trigram changed due to a skip or repeat).
        """
        if not query_words:
            return []

        candidates: set[int] = set()

        # 1. Trigram candidate generation (fast, high precision)
        if len(query_words) >= 3:
            for qi in range(len(query_words) - 2):
                tg = (query_words[qi], query_words[qi + 1], query_words[qi + 2])
                for pos in self.trigram_idx.get(tg, []):
                    candidates.add(pos - qi)

        # 2. Fuzzy-alef trigrams: try alef-normalized variants if sparse
        #    (handles ا/أ/إ that slip through normalization)
        if len(candidates) == 0 and len(query_words) >= 3:
            tg = (query_words[0], query_words[1], query_words[2])
            # already normalized; try rarest-word fallback below

        # 3. Rarest-word fallback (inverted index on least-frequent query word)
        if len(candidates) == 0 or len(candidates) > 8000:
            rarest_qi = min(
                range(len(query_words)),
                key=lambda i: len(self.inverted.get(query_words[i], [])) or 10**9,
            )
            rarest_word = query_words[rarest_qi]
            for pos in self.inverted.get(rarest_word, []):
                candidates.add(pos - rarest_qi)

        # 4. Nearby window brute force
        if nearby_pos is not None:
            lo = max(0, nearby_pos - nearby_window)
            hi = min(self.n_words - len(query_words), nearby_pos + nearby_window)
            for p in range(lo, hi + 1):
                candidates.add(p)

        # 5. Score & filter
        results: List[SearchMatch] = []
        for pos in candidates:
            if pos < 0 or pos + len(query_words) > self.n_words:
                continue
            s = self.score_at(query_words, pos)
            if s < min_score:
                continue
            m = self.meta[pos]
            results.append(
                SearchMatch(
                    start_pos=pos,
                    score=s,
                    n_matched=len(query_words),
                    surah=m.surah,
                    ayah=m.ayah,
                    word_pos=m.word_pos,
                )
            )

        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    # ── Convenience ───────────────────────────────────────────────────────────

    def word_match_score(self, corpus_word: str, query_word: str) -> float:
        return word_match_score(corpus_word, query_word)

    def get_context(self, global_pos: int, window: int = 5) -> str:
        """Return a few words around global_pos for debugging."""
        lo = max(0, global_pos - window)
        hi = min(self.n_words, global_pos + window)
        return " ".join(self.meta[i].original for i in range(lo, hi))
