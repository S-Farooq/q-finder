"""
Real-time Quran position tracker.

State machine:
  SEARCHING  ──(≥3 final tokens with score ≥ LOCK_SCORE)──►  TRACKING
  TRACKING   ──(confidence drops below UNLOCK_SCORE)      ──►  SEARCHING

Beam search:
  - Maintains up to BEAM_SIZE position hypotheses.
  - Each hypothesis tracks a rolling window of recent word-match scores.
  - The best hypothesis determines the reported position.
  - New hypotheses are seeded from global search every SEARCH_EVERY tokens
    or immediately when the beam collapses.

Switch detection:
  - When tracking is lost and a new lock is found that is in a different surah
    OR more than SWITCH_AYAH_DISTANCE ayahs away from where we last had
    high-confidence tracking, a "switch" event is emitted.

Non-final tokens:
  - Used only for speculative/preview output; they never change the beam state.
  - The server can use the speculative result for a low-latency display.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional

from .index import QuranIndex, word_match_score
from .normalize import normalize_word, tokenize

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

BEAM_SIZE             = 8
SCORE_WINDOW          = 8     # rolling window size for confidence
MATCH_REWARD          = 0.22  # added to window on a perfect match
LOCK_SCORE            = 0.80  # search result score needed to lock (high-precision)
UNLOCK_CONFIDENCE     = 0.28  # beam confidence below this → back to SEARCHING
SEARCH_EVERY          = 4     # seed new global search every N final tokens
MIN_TOKENS_TO_SEARCH  = 4     # need this many final tokens before searching
QUERY_WINDOW          = 6     # last N final tokens used as search query
SWITCH_AYAH_DISTANCE  = 30    # ayah distance (same-surah) that counts as a switch

# Ambiguity / uniqueness gate (precision-first locking)
# We only commit to a position when the best match is clearly better than
# the second-best.  If two locations score similarly the phrase is not
# unique enough to trust — we stay SEARCHING until we accumulate more words.
MIN_SCORE_GAP        = 0.15   # best − second must exceed this to lock
AMBIGUITY_MIN_SCORE  = 0.65   # second candidate must be above this to trigger the check


# ── Data classes ──────────────────────────────────────────────────────────────

class TrackerState(str, Enum):
    SEARCHING = "searching"
    TRACKING  = "tracking"


@dataclass
class Hypothesis:
    """One beam candidate."""
    next_word_pos: int               # global index of the word we expect next
    scores: Deque[float] = field(default_factory=lambda: deque(maxlen=SCORE_WINDOW))

    @property
    def confidence(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def push_score(self, s: float) -> None:
        self.scores.append(s)

    def clone(self) -> "Hypothesis":
        h = Hypothesis(next_word_pos=self.next_word_pos)
        h.scores = deque(self.scores, maxlen=SCORE_WINDOW)
        return h


@dataclass
class TrackingResult:
    state:       str
    surah:       Optional[int]
    ayah:        Optional[int]
    word_pos:    Optional[int]  # 0-based word position within the ayah
    confidence:  float
    is_switch:   bool                  = False
    switch_from: Optional[dict]        = None   # {surah, ayah} of last confirmed pos
    # extra context for the API response
    surah_name:  Optional[str]         = None
    ayah_text:   Optional[str]         = None


# ── Tracker ───────────────────────────────────────────────────────────────────

class QuranTracker:
    def __init__(
        self,
        index: QuranIndex,
        beam_size:             int   = BEAM_SIZE,
        lock_score:            float = LOCK_SCORE,
        unlock_confidence:     float = UNLOCK_CONFIDENCE,
        search_every:          int   = SEARCH_EVERY,
        min_tokens_to_search:  int   = MIN_TOKENS_TO_SEARCH,
        query_window:          int   = QUERY_WINDOW,
        switch_ayah_distance:  int   = SWITCH_AYAH_DISTANCE,
        min_score_gap:         float = MIN_SCORE_GAP,
        ambiguity_min_score:   float = AMBIGUITY_MIN_SCORE,
    ) -> None:
        self.index                 = index
        self.beam_size             = beam_size
        self.lock_score            = lock_score
        self.unlock_confidence     = unlock_confidence
        self.search_every          = search_every
        self.min_tokens_to_search  = min_tokens_to_search
        self.query_window          = query_window
        self.switch_ayah_distance  = switch_ayah_distance
        self.min_score_gap         = min_score_gap
        self.ambiguity_min_score   = ambiguity_min_score

        self._reset_state()

    def _reset_state(self) -> None:
        self.state:             TrackerState    = TrackerState.SEARCHING
        self.beam:              List[Hypothesis] = []
        self.final_buffer:      List[str]        = []   # last N final tokens
        self.nonfinal_buffer:   List[str]        = []   # current non-final run
        self.token_count:       int              = 0    # total final tokens seen
        self.current_pos:       Optional[int]    = None # best global word pos (next expected)
        self.last_lock_pos:     Optional[int]    = None # global pos when last confident
        self.last_lock_surah:   Optional[int]    = None
        self.last_lock_ayah:    Optional[int]    = None

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Full reset — call when a new session starts."""
        self._reset_state()

    def process_token(self, text: str, is_final: bool) -> Optional[TrackingResult]:
        """
        Process one token from the STT stream.

        *text* may be:
          • A single assembled word         (final or non-final single token)
          • A space-separated segment text  (non-final full-segment hypothesis
                                             from Soniox — replaces the buffer)

        Returns a TrackingResult on meaningful state/position changes.
        """
        if not is_final:
            # ── Non-final path ────────────────────────────────────────────────
            if " " in text:
                # Full segment hypothesis — replace buffer entirely so the
                # lookahead always reflects what Soniox thinks right now.
                self.nonfinal_buffer = [
                    w for w in (normalize_word(t) for t in text.split()) if w
                ][-8:]
            else:
                norm = normalize_word(text)
                if not norm:
                    return None
                self.nonfinal_buffer.append(norm)
                if len(self.nonfinal_buffer) > 8:
                    self.nonfinal_buffer.pop(0)

            if self.state == TrackerState.TRACKING and self.beam:
                return self._nonfinal_lookahead()
            else:
                return self._speculative_search()

        # ── Final token path ──────────────────────────────────────────────────
        self.nonfinal_buffer.clear()
        norm = normalize_word(text)
        if not norm:
            return None
        self.final_buffer.append(norm)
        if len(self.final_buffer) > self.query_window * 2:
            self.final_buffer.pop(0)
        self.token_count += 1

        if self.state == TrackerState.TRACKING:
            return self._advance_beam(norm)
        else:
            return self._do_search()

    def _nonfinal_lookahead(self) -> Optional[TrackingResult]:
        """
        Walk the non-final buffer forward from the committed beam position.

        Because STT finalises in batches, the reciter may be several words
        ahead of the last final token.  We try multiple start offsets (0..8
        words past committed pos) and pick the one that yields the longest
        clean consecutive match run.
        """
        if not self.beam or not self.nonfinal_buffer:
            return None

        committed = self.beam[0].next_word_pos  # next expected word (from finals)
        committed_conf = self.beam[0].confidence

        best_end   = committed   # furthest position reached
        best_score = 0.0

        # Try start offsets 0 .. min(8, nf_len) words past committed pos.
        # Offset 0: finals are up-to-date, nonfinals start right at committed.
        # Offset N: finals are N words behind the live speech.
        max_offset = min(8, len(self.nonfinal_buffer))
        for start_offset in range(0, max_offset + 1):
            start   = committed + start_offset
            matched = 0
            total_s = 0.0
            for i, tok in enumerate(self.nonfinal_buffer):
                p = start + i
                if p >= self.index.n_words:
                    break
                s = word_match_score(self.index.words[p], tok)
                if s < 0.5:
                    break               # stop on first poor match
                total_s += s
                matched  += 1

            if matched == 0:
                continue
            end_pos = start + matched
            avg     = total_s / matched

            # Prefer the furthest consistent end position; break ties by score
            if end_pos > best_end or (end_pos == best_end and avg > best_score):
                best_end   = end_pos
                best_score = avg

        if best_end <= committed or best_end == 0:
            return None

        meta = self.index.meta[best_end - 1]
        return TrackingResult(
            state="tracking_speculative",
            surah=meta.surah,
            ayah=meta.ayah,
            word_pos=meta.word_pos,
            confidence=round(committed_conf * best_score, 3),
        )

    def _speculative_search(self) -> Optional[TrackingResult]:
        """
        While still SEARCHING (no final lock yet), attempt a phrase search
        using the last few final tokens combined with the current non-final
        buffer.  Returns a low-confidence speculative hint, or None if
        the phrase is ambiguous / not found.

        This lets the UI show "you might be at surah X ayah Y" before we
        have enough final tokens to commit.
        """
        combined = self.final_buffer[-4:] + list(self.nonfinal_buffer[-4:])
        if len(combined) < self.min_tokens_to_search:
            return None

        matches = self.index.search(combined, top_k=5, min_score=self.lock_score)
        if not matches:
            return None

        best_score   = matches[0].score
        second_score = matches[1].score if len(matches) > 1 else 0.0
        gap          = best_score - second_score
        if second_score >= self.ambiguity_min_score and gap < self.min_score_gap:
            return None    # ambiguous — don't hint

        m = matches[0]
        return TrackingResult(
            state="searching_speculative",
            surah=m.surah,
            ayah=m.ayah,
            word_pos=m.word_pos,
            confidence=round(m.score * 0.65, 3),   # discounted — not yet final
        )

    # ── Beam advance ──────────────────────────────────────────────────────────

    def _advance_beam(self, new_token: str) -> TrackingResult:
        """Update beam with a new final token and return best position."""
        new_beam: List[Hypothesis] = []

        for hyp in self.beam:
            p = hyp.next_word_pos

            # Try exact position
            s0 = word_match_score(self.index.words[p], new_token) if p < self.index.n_words else 0.0

            # Try +1 skip (speaker skipped one word)
            s1 = word_match_score(self.index.words[p + 1], new_token) if p + 1 < self.index.n_words else 0.0

            # Try -1 repeat (speaker repeated)
            sr = word_match_score(self.index.words[p - 1], new_token) if p - 1 >= 0 else 0.0

            best_s   = max(s0, s1, sr)
            best_adv = 1 if s0 >= max(s1, sr) else (2 if s1 > sr else 0)

            if best_s > 0:
                h2 = hyp.clone()
                h2.push_score(best_s)
                h2.next_word_pos = p + best_adv
                new_beam.append(h2)
            else:
                # Mismatch: penalise but keep hypothesis alive
                h2 = hyp.clone()
                h2.push_score(0.0)
                if h2.confidence > self.unlock_confidence / 2:
                    new_beam.append(h2)

        # Periodically seed new hypotheses from global search
        if self.token_count % self.search_every == 0:
            new_beam.extend(self._search_hypotheses(nearby=True))

        # Sort and prune
        new_beam.sort(key=lambda h: -h.confidence)
        self.beam = new_beam[: self.beam_size]

        # Check if we lost tracking
        if not self.beam or self.beam[0].confidence < self.unlock_confidence:
            return self._lose_tracking()

        best = self.beam[0]
        self.current_pos = best.next_word_pos

        # Update "last lock" position for switch detection
        if best.confidence > 0.65 and self.current_pos > 0:
            meta = self.index.meta[self.current_pos - 1]
            self.last_lock_pos   = self.current_pos
            self.last_lock_surah = meta.surah
            self.last_lock_ayah  = meta.ayah

        return self._make_result(best)

    def _lose_tracking(self) -> TrackingResult:
        """Called when beam collapses below threshold."""
        logger.debug("Lost tracking at pos=%s", self.current_pos)
        self.state       = TrackerState.SEARCHING
        self.beam        = []
        self.current_pos = None
        result = self._do_search()
        return result or TrackingResult(
            state="searching", surah=None, ayah=None,
            word_pos=None, confidence=0.0,
        )

    # ── Global search ─────────────────────────────────────────────────────────

    def _do_search(self) -> Optional[TrackingResult]:
        """Run phrase search with the current final token buffer."""
        if len(self.final_buffer) < self.min_tokens_to_search:
            return TrackingResult(
                state="searching", surah=None, ayah=None,
                word_pos=None, confidence=0.0,
            )

        query = self.final_buffer[-self.query_window:]

        # Try full query first, then shorter if no good match
        for q_len in [len(query), max(self.min_tokens_to_search, len(query) - 2)]:
            q = query[-q_len:]
            nearby = self.current_pos  # may be None
            matches = self.index.search(
                q,
                top_k=10,
                nearby_pos=nearby,
                nearby_window=200,
                min_score=self.lock_score,
            )
            if matches:
                break

        if not matches:
            return TrackingResult(
                state="searching", surah=None, ayah=None,
                word_pos=None, confidence=0.0,
            )

        # ── Uniqueness / ambiguity gate ───────────────────────────────────────
        # Only commit to a position when the best candidate is clearly ahead of
        # the second-best.  If the phrase appears in multiple locations with
        # similar scores we do NOT lock — we stay in SEARCHING and wait for
        # more tokens that disambiguate (e.g. the next unique word).
        best_score   = matches[0].score
        second_score = matches[1].score if len(matches) > 1 else 0.0
        gap          = best_score - second_score

        if second_score >= self.ambiguity_min_score and gap < self.min_score_gap:
            logger.debug(
                "Ambiguous phrase (best=%.2f second=%.2f gap=%.2f) — "
                "waiting for more context", best_score, second_score, gap,
            )
            return TrackingResult(
                state="searching", surah=None, ayah=None,
                word_pos=None, confidence=0.0,
            )

        # Lock onto best match
        best_match = matches[0]
        lock_pos   = best_match.start_pos + len(q)

        # Build beam from top matches
        self.beam = []
        for m in matches[: self.beam_size]:
            h = Hypothesis(next_word_pos=m.start_pos + len(q))
            # Seed the rolling window with the search score so confidence
            # starts meaningfully high
            for _ in range(min(4, SCORE_WINDOW)):
                h.push_score(m.score)
            self.beam.append(h)

        self.state       = TrackerState.TRACKING
        self.current_pos = lock_pos

        # Switch detection
        is_switch   = False
        switch_from = None
        if self.last_lock_surah is not None:
            if best_match.surah != self.last_lock_surah:
                is_switch   = True
                switch_from = {"surah": self.last_lock_surah, "ayah": self.last_lock_ayah}
            elif abs(best_match.ayah - self.last_lock_ayah) > self.switch_ayah_distance:
                is_switch   = True
                switch_from = {"surah": self.last_lock_surah, "ayah": self.last_lock_ayah}

        logger.info(
            "Locked: surah=%d ayah=%d score=%.2f switch=%s",
            best_match.surah, best_match.ayah, best_match.score, is_switch,
        )

        return TrackingResult(
            state="tracking",
            surah=best_match.surah,
            ayah=best_match.ayah,
            word_pos=best_match.word_pos,
            confidence=best_match.score,
            is_switch=is_switch,
            switch_from=switch_from,
        )

    def _search_hypotheses(self, nearby: bool = False) -> List[Hypothesis]:
        """Return new Hypothesis objects from a search, for beam seeding."""
        if len(self.final_buffer) < self.min_tokens_to_search:
            return []
        q = self.final_buffer[-min(self.query_window, len(self.final_buffer)):]
        matches = self.index.search(
            q,
            top_k=3,
            nearby_pos=self.current_pos if nearby else None,
            nearby_window=100,
            min_score=self.lock_score - 0.1,
        )
        hyps = []
        for m in matches:
            h = Hypothesis(next_word_pos=m.start_pos + len(q))
            h.push_score(m.score * 0.8)   # slightly discounted — just a candidate
            hyps.append(h)
        return hyps

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_result(self, hyp: Hypothesis) -> TrackingResult:
        pos = hyp.next_word_pos - 1
        if pos < 0 or pos >= self.index.n_words:
            return TrackingResult(
                state="tracking", surah=None, ayah=None,
                word_pos=None, confidence=hyp.confidence,
            )
        meta = self.index.meta[pos]
        return TrackingResult(
            state="tracking",
            surah=meta.surah,
            ayah=meta.ayah,
            word_pos=meta.word_pos,
            confidence=round(hyp.confidence, 3),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def best_hypothesis(self) -> Optional[Hypothesis]:
        return self.beam[0] if self.beam else None

    def debug_beam(self) -> str:
        lines = []
        for i, h in enumerate(self.beam[:4]):
            pos = h.next_word_pos
            ctx = self.index.get_context(pos, window=3) if pos < self.index.n_words else "?"
            lines.append(f"  [{i}] pos={pos} conf={h.confidence:.2f} | {ctx}")
        return "\n".join(lines)
