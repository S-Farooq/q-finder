"""
FastAPI WebSocket server for real-time Quran tracking.

Start:
    uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
    SONIOX_API_KEY   — required, your Soniox API key
    INDEX_PATH       — path to pre-built pickle index (default: quran_index.pkl)
    DATA_PATH        — path to hafs_smart_v8.json   (default: hafs_smart_v8.json)
    SONIOX_MODEL     — Soniox model name             (default: stt-rt-preview)
    SONIOX_LANGUAGE  — language hint                 (default: ar)
    SONIOX_SAMPLE_RATE — audio sample rate in Hz     (default: 16000)
    LOG_LEVEL        — Python logging level          (default: INFO)

WebSocket protocol  (endpoint: ws://<host>:8000/track)
─────────────────────────────────────────────────────────────────────────────
Client → Server
  Binary frames: raw 16-bit LE PCM audio chunks (mono, 16 kHz default).
  Text frames:   JSON control messages
    {"type": "reset"}          — reset tracker state for this session
    {"type": "config", ...}    — reserved for future per-session config

Server → Client  (JSON text frames)
  Tracking update:
    {
      "type":        "position",
      "state":       one of:
                       "searching"            — no lock yet, accumulating context
                       "searching_speculative" — unconfirmed hint from non-final tokens
                       "tracking"             — committed lock (use for definitive display)
                       "tracking_speculative" — live ahead-of-final estimate (low-latency)
      "surah":       <int 1–114 or null>,
      "ayah":        <int or null>,
      "word_pos":    <int 0-based within ayah, or null>,
      "confidence":  <float 0–1>,
      "is_switch":   <bool>,
      "switch_from": <{"surah": int, "ayah": int} or null>
    }
  Latency note:
    "tracking_speculative" events typically lead "tracking" by 1–3 seconds.
    They reflect the current non-final STT hypothesis projected forward from
    the last confirmed position.  Show them as a live preview; use "tracking"
    events for recording or navigation.
  Error:
    {"type": "error", "message": "<description>"}
  Ready:
    {"type": "ready", "words_indexed": <int>}
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from quran_tracker.index import QuranIndex
from quran_tracker.tracker import QuranTracker, TrackingResult
from quran_tracker.soniox_client import SonioxStreamer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH    = Path(os.environ.get("INDEX_PATH",    "quran_index.pkl"))
DATA_PATH     = Path(os.environ.get("DATA_PATH",     "hafs_smart_v8.json"))
SONIOX_MODEL  = os.environ.get("SONIOX_MODEL",       "stt-rt-preview")
SONIOX_LANG   = os.environ.get("SONIOX_LANGUAGE",    "ar")
SONIOX_RATE   = int(os.environ.get("SONIOX_SAMPLE_RATE", "16000"))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quran Real-Time Tracker",
    description="Stream Arabic audio → get Quran position in real time",
)

_quran_index: Optional[QuranIndex] = None


@app.on_event("startup")
async def _startup() -> None:
    global _quran_index
    if INDEX_PATH.exists():
        logger.info("Loading pre-built index from %s …", INDEX_PATH)
        _quran_index = await asyncio.get_event_loop().run_in_executor(
            None, QuranIndex.load, str(INDEX_PATH)
        )
    else:
        logger.info("Building index from %s (first run, ~5–10 s) …", DATA_PATH)
        idx = QuranIndex()
        await asyncio.get_event_loop().run_in_executor(
            None, idx.build_from_json, str(DATA_PATH)
        )
        await asyncio.get_event_loop().run_in_executor(
            None, idx.save, str(INDEX_PATH)
        )
        _quran_index = idx
    logger.info("Index ready: %d words", _quran_index.n_words)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def frontend():
    from fastapi.responses import FileResponse
    if Path("frontend.html").exists():
        return FileResponse("frontend.html")
    return JSONResponse({
        "message": "Quran Tracker API",
        "docs": "/docs",
        "health": "/health",
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "words_indexed": _quran_index.n_words if _quran_index else 0,
    })


@app.get("/verse/{surah}/{ayah}")
async def verse_endpoint(surah: int, ayah: int) -> JSONResponse:
    """Return the original words of a single ayah for display."""
    if not _quran_index:
        return JSONResponse({"error": "index not ready"}, status_code=503)
    words = _quran_index.get_ayah_words(surah, ayah)
    if not words:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"surah": surah, "ayah": ayah, "words": words})


@app.get("/search")
async def search_endpoint(q: str, top_k: int = 5) -> JSONResponse:
    """
    Quick phrase search for debugging.
    ?q=<arabic text>&top_k=5
    """
    if not _quran_index:
        return JSONResponse({"error": "index not ready"}, status_code=503)
    from quran_tracker.normalize import tokenize
    query_words = tokenize(q)
    matches = _quran_index.search(query_words, top_k=top_k)
    return JSONResponse([
        {
            "surah":    m.surah,
            "ayah":     m.ayah,
            "word_pos": m.word_pos,
            "score":    round(m.score, 3),
            "context":  _quran_index.get_context(m.start_pos, window=5),
        }
        for m in matches
    ])


# ── WebSocket tracking endpoint ───────────────────────────────────────────────

@app.websocket("/track")
async def track_ws(ws: WebSocket) -> None:
    await ws.accept()

    if not _quran_index:
        await ws.send_json({"type": "error", "message": "index not ready"})
        await ws.close()
        return

    tracker   = QuranTracker(_quran_index)
    audio_q:  asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=256)
    # result_q carries both raw token dicts and TrackingResult objects
    result_q: asyncio.Queue = asyncio.Queue(maxsize=512)

    # Send ready signal
    await ws.send_json({"type": "ready", "words_indexed": _quran_index.n_words})

    # ── Token callback (called by SonioxStreamer) ──────────────────────────────

    async def _on_token(text: str, is_final: bool) -> None:
        # Forward raw STT token to client for debug display
        await result_q.put({"type": "token", "text": text, "is_final": is_final})
        # Run through tracker; may produce a position update
        result = tracker.process_token(text, is_final)
        if result is not None:
            await result_q.put(result)

    # ── Soniox streaming task ─────────────────────────────────────────────────

    async def _audio_source():
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                return
            yield chunk

    streamer = SonioxStreamer(
        token_callback   = _on_token,
        model            = SONIOX_MODEL,
        language         = SONIOX_LANG,
        sample_rate      = SONIOX_RATE,
        include_nonfinal = True,
    )

    async def _soniox_task() -> None:
        try:
            await streamer.run(_audio_source())
        except Exception as exc:
            logger.exception("Soniox error: %s", exc)
            await result_q.put(None)   # signal done

    soniox_fut = asyncio.create_task(_soniox_task())

    # ── Position broadcast task ───────────────────────────────────────────────

    async def _broadcast_task() -> None:
        while True:
            item = await result_q.get()
            if item is None:
                break
            try:
                if isinstance(item, dict):
                    await ws.send_json(item)
                else:
                    await ws.send_json(_result_to_dict(item))
            except Exception:
                break

    broadcast_fut = asyncio.create_task(_broadcast_task())

    # ── Receive audio / control from client ───────────────────────────────────
    # Handles both text frames (JSON control) and binary frames (PCM audio).

    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break

            if frame.get("text"):
                # JSON control message sent as a text WebSocket frame
                try:
                    ctrl = json.loads(frame["text"])
                    if ctrl.get("type") == "reset":
                        tracker.reset()
                        logger.info("Tracker reset by client")
                except json.JSONDecodeError:
                    pass

            elif frame.get("bytes"):
                # Raw PCM audio sent as a binary WebSocket frame
                await audio_q.put(frame["bytes"])

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:
        logger.exception("WS error: %s", exc)
    finally:
        await audio_q.put(None)   # stop audio source → Soniox finalise
        await asyncio.wait_for(soniox_fut, timeout=5.0)
        await result_q.put(None)
        await broadcast_fut
        logger.info("Session closed, beam debug:\n%s", tracker.debug_beam())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_to_dict(r: TrackingResult) -> dict:
    return {
        "type":        "position",
        "state":       r.state,
        "surah":       r.surah,
        "ayah":        r.ayah,
        "word_pos":    r.word_pos,
        "confidence":  round(r.confidence, 3),
        "is_switch":   r.is_switch,
        "switch_from": r.switch_from,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
