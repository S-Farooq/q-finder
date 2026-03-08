"""
Soniox real-time STT client — SDK 2.x (tested v2.2.0).

Key design decisions
─────────────────────────────────────────────────────────────────────────────
1. Sub-word reconstruction
   Soniox tokens use leading spaces to mark word boundaries — a token that
   starts with a space is the beginning of a new word; tokens without a
   leading space are sub-word continuations of the previous token.
   e.g. المغضوب → ["الم", "غ", "ض", " وب"] — the space on " وب" is absent
   so we concatenate until we see a new leading-space token.
   We simply join all token texts and split on whitespace.

2. Non-final deduplication
   Each Soniox event re-emits the FULL current-segment hypothesis as non-final
   tokens — the same words appear in every event until they are finalized.
   We emit ONE combined non-final text per event (not one row per token) so
   the tracker and the UI can treat it as a live caption that gets replaced.

3. Graceful shutdown
   When the client disconnects the mic, Soniox times out (408) and closes the
   WebSocket.  We catch the resulting error in _send so it doesn't bubble.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import AsyncGenerator, Callable, Optional

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str, bool], Optional[object]]

# Strip Arabic and Latin punctuation from token text
_PUNCT_RE = re.compile(r"[،؟؛!.,;:\-\"\'()«»\[\]{}]")

# Soniox special marker tokens like <end>, <unk> etc.
_SPECIAL_RE = re.compile(r"^<[^>]+>$")


def _is_special(token_text: str) -> bool:
    return bool(_SPECIAL_RE.match(token_text.strip()))


def _clean(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip()




class SonioxStreamer:
    """
    Wraps the Soniox async realtime STT session and dispatches tokens to
    *token_callback(text, is_final)*.

    Final tokens   → assembled into words, called once per word.
    Non-final text → the FULL current-segment hypothesis is joined into one
                     string and called once per event (not once per token).
                     The tracker treats a space-containing non-final as a full
                     segment replacement rather than an incremental append.
    """

    def __init__(
        self,
        token_callback:   TokenCallback,
        api_key:          Optional[str] = None,
        model:            str           = "stt-rt-preview",
        language:         str           = "ar",
        sample_rate:      int           = 16_000,
        include_nonfinal: bool          = True,
    ) -> None:
        self.token_callback   = token_callback
        self.api_key          = api_key or os.environ.get("SONIOX_API_KEY", "")
        self.model            = model
        self.language         = language
        self.sample_rate      = sample_rate
        self.include_nonfinal = include_nonfinal
        # Stable-prefix promotion state
        self._prev_nf_words: list[str] = []   # words from last non-final event
        self._nf_promoted: int = 0             # how many words in _prev_nf_words already dispatched
        self._held_word: Optional[str] = None  # last stable word held back (may be partial)
        self._hold_timer: Optional[asyncio.Task] = None

    async def run(self, audio_source: AsyncGenerator[bytes, None]) -> None:
        from soniox.client import AsyncSonioxClient          # type: ignore[import]
        from soniox.types.realtime import RealtimeSTTConfig  # type: ignore[import]

        config = RealtimeSTTConfig(
            model                 = self.model,
            audio_format          = "pcm_s16le",
            sample_rate           = self.sample_rate,
            num_channels          = 1,
            language_hints        = [self.language],
            language_hints_strict = True,
        )

        client = AsyncSonioxClient(api_key=self.api_key)

        async with client.realtime.stt.connect(config=config) as session:
            await asyncio.gather(
                self._send(session, audio_source),
                self._receive(session),
            )

    # ── Audio sender ──────────────────────────────────────────────────────────

    async def _send(self, session, audio_source: AsyncGenerator[bytes, None]) -> None:
        try:
            async for chunk in audio_source:
                await session.send_byte_chunk(chunk)
            await session.finish()
        except Exception as exc:
            # Soniox may close the session first (408 timeout, client disconnect)
            logger.debug("Soniox send finished early: %s", exc)

    # ── Event receiver ────────────────────────────────────────────────────────

    async def _receive(self, session) -> None:
        """
        Process events from Soniox.

        We ignore actual final tokens for word dispatching (they are far too
        delayed). Instead we promote words from the non-final hypothesis the
        moment they are *stable* — i.e., appear in the same position across
        two consecutive non-final events. A held-back word (the last stable
        word, which might still be mid-build) is flushed immediately when:
          - The text grows (a new word appears after it), or
          - The text shifts left (Soniox internalized earlier words), or
          - A 500 ms hold timer fires with no change.

        Actual final tokens are only inspected for the <end> segment marker,
        which resets the promotion state for the new segment.
        """
        final_count = 0

        async for event in session.receive_events():
            if event.error_code:
                logger.warning("Soniox: %s — %s", event.error_code, event.error_message)
                if event.error_code != 408:
                    break
                continue

            # ── Watch for <end> to reset segment state ────────────────────────
            all_finals = [t for t in event.tokens if t.is_final]
            new_finals = all_finals[final_count:]
            final_count += len(new_finals)
            if any(_is_special(t.text or "") for t in new_finals):
                logger.debug("Soniox <end> — resetting non-final promotion state")
                self._reset_nf_state()
                final_count = 0

            # ── Non-final: promote stable words, send tail as caption ─────────
            if self.include_nonfinal:
                nonfinals = [t for t in event.tokens if not t.is_final]
                if nonfinals:
                    raw = "".join(t.text or "" for t in nonfinals)
                    caption = await self._promote_stable(raw)
                    if caption:
                        await self._dispatch(caption, False)

    # ── Stable-prefix promotion ───────────────────────────────────────────────

    async def _promote_stable(self, text: str) -> str:
        """
        Compare the new non-final text against the previous event.
        Dispatch words that are stable (same position, two events in a row)
        as is_final=True.  Returns the unconfirmed tail for the live caption.

        Three cases on each event:
          A. Stable prefix GREW     → new confirmed words available; flush held,
                                      dispatch, hold back new last stable word.
          B. No stable prefix (=0)  → text shifted left; flush held immediately,
                                      reset window for next event.
          C. Same stable prefix     → word at the hold position may still be
                                      building (نستع→نستعين); update held word
                                      and reset the hold timer.
        """
        curr = [w for w in (_clean(p) for p in text.split()) if w and not _is_special(w)]
        if not curr:
            return ""

        prev = self._prev_nf_words
        self._prev_nf_words = curr

        if not prev:
            return " ".join(curr)

        # Length of the longest common prefix between prev and curr
        stable_len = sum(
            1 for _ in
            (None for a, b in zip(prev, curr) if a == b)
        )
        # (re-compute properly since the generator above doesn't short-circuit)
        stable_len = 0
        for a, b in zip(prev, curr):
            if a == b:
                stable_len += 1
            else:
                break

        wp = self._nf_promoted

        if stable_len > wp:
            # Case A: new stable words
            await self._flush_held()
            wp = self._nf_promoted  # refresh after flush

            # Hold back the last stable word unless curr grew beyond it
            promote_end = stable_len if len(curr) > stable_len else stable_len - 1

            for word in curr[wp:promote_end]:
                if len(word) > 1:   # skip single-char sub-word fragments
                    logger.debug("Soniox stable→final: %r", word)
                    await self._dispatch(word, True)
            self._nf_promoted = promote_end

            if promote_end < stable_len:
                self._held_word = curr[promote_end]
                self._arm_hold_timer()
            else:
                self._held_word = None

        elif stable_len == 0:
            # Case B: text shifted left — flush held, reset window
            await self._flush_held()
            self._nf_promoted = 0

        else:
            # Case C: same stable prefix; word at wp position may still be building
            if self._held_word and len(curr) > wp:
                new_word = curr[wp]
                if new_word != self._held_word:
                    if new_word.startswith(self._held_word):
                        # Word is still growing: update and reset timer
                        self._held_word = new_word
                        self._arm_hold_timer()
                    # else: hypothesis changed; let timer handle it

        return " ".join(curr[self._nf_promoted:])

    # ── Hold-timer helpers ────────────────────────────────────────────────────

    def _arm_hold_timer(self, delay: float = 0.5) -> None:
        if self._hold_timer and not self._hold_timer.done():
            self._hold_timer.cancel()
        self._hold_timer = asyncio.create_task(self._delayed_hold_flush(delay))

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer and not self._hold_timer.done():
            self._hold_timer.cancel()
        self._hold_timer = None

    async def _delayed_hold_flush(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._flush_held()

    async def _flush_held(self) -> None:
        """Dispatch the held-back word and advance the promotion counter."""
        self._cancel_hold_timer()
        if self._held_word:
            word = self._held_word
            self._held_word = None
            self._nf_promoted += 1
            if len(word) > 1:
                logger.debug("Soniox held→final: %r", word)
                await self._dispatch(word, True)

    def _reset_nf_state(self) -> None:
        """Reset all promotion state at segment boundary (<end>)."""
        self._cancel_hold_timer()
        self._prev_nf_words = []
        self._nf_promoted = 0
        self._held_word = None

    async def _dispatch(self, text: str, is_final: bool) -> None:
        result = self.token_callback(text, is_final)
        if asyncio.iscoroutine(result):
            await result


# ── Sync helper for offline / CLI testing ─────────────────────────────────────

def run_sync(
    audio_path:     str,
    token_callback: TokenCallback,
    api_key:        Optional[str] = None,
    model:          str           = "stt-rt-preview",
    language:       str           = "ar",
    chunk_ms:       int           = 100,
    sample_rate:    int           = 16_000,
) -> None:
    """
    Blocking helper: read a raw 16-bit PCM file and stream it through Soniox.
    audio_path — raw 16-bit LE mono PCM at *sample_rate* Hz.
    """
    bytes_per_ms = sample_rate * 2 // 1000
    chunk_size   = bytes_per_ms * chunk_ms

    async def _file_source() -> AsyncGenerator[bytes, None]:
        with open(audio_path, "rb") as fh:
            while True:
                data = fh.read(chunk_size)
                if not data:
                    break
                yield data
                await asyncio.sleep(0)

    asyncio.run(SonioxStreamer(
        token_callback = token_callback,
        api_key        = api_key,
        model          = model,
        language       = language,
        sample_rate    = sample_rate,
    ).run(_file_source()))
