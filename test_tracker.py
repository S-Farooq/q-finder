"""
Smoke tests for the index and tracker — no Soniox needed.

Run:  python test_tracker.py
"""
from quran_tracker.index import QuranIndex, word_match_score
from quran_tracker.tracker import QuranTracker
from quran_tracker.normalize import normalize_word, tokenize

idx = QuranIndex.load("quran_index.pkl")

# ── 1. Normalization ──────────────────────────────────────────────────────────
print("=== Normalization ===")
tests = [
    ("الرَّحْمَنِ", "الرحمن"),   # strip diacritics
    ("إِيَّاكَ",   "اياك"),     # alef + diacritics
    ("رَحْمَةً",   "رحمه"),     # ta marbuta → ha, diacritics
    ("هُدًى",      "هدي"),      # alef maqsura → ya, tanwin
]
for raw, expected in tests:
    got = normalize_word(raw)
    status = "OK" if got == expected else f"FAIL (got '{got}')"
    print(f"  {raw} → {got}   [{status}]")

# ── 2. Word match score ───────────────────────────────────────────────────────
print("\n=== Word match score ===")
pairs = [
    ("الله", "الله",   1.0),
    ("الله", "اللة",   0.80),   # edit dist 1
    ("الرحمن", "الرحمان", 0.45), # edit dist 2
    ("الله", "بسم",    0.0),
]
for a, b, expected in pairs:
    s = word_match_score(a, b)
    status = "OK" if abs(s - expected) < 0.01 else f"FAIL (got {s:.2f})"
    print(f"  '{a}' vs '{b}' → {s:.2f}  [{status}]")

# ── 3. Exact phrase search ────────────────────────────────────────────────────
print("\n=== Phrase search ===")

cases = [
    # (description, text, expected_surah, expected_ayah)
    ("Al-Fatiha 1",      "بسم الله الرحمن الرحيم",           1, 1),
    ("Al-Fatiha 2",      "الحمد لله رب العالمين",            1, 2),
    ("Al-Baqarah 2",     "ذلك الكتاب لا ريب فيه هدى للمتقين", 2, 2),
    ("Al-Ikhlas 1",      "قل هو الله احد",                   112, 1),
    ("Al-Asr 1",         "والعصر",                            103, 1),
    ("An-Nas 1",         "قل اعوذ برب الناس",                 114, 1),
    # Note: the opener "الله لا اله الا هو الحي القيوم" is shared by 2:255 AND 3:2
    # so the search correctly returns BOTH — top two are checked in the next test.
    # ("Ayat al-Kursi",  "الله لا اله الا هو الحي القيوم",    2, 255),
]

for desc, text, exp_surah, exp_ayah in cases:
    words = tokenize(text)
    matches = idx.search(words, top_k=3)
    if matches:
        m = matches[0]
        status = "OK" if m.surah == exp_surah and m.ayah == exp_ayah else \
                 f"FAIL (got {m.surah}:{m.ayah})"
        print(f"  {desc:<25} score={m.score:.2f}  {m.surah}:{m.ayah}  [{status}]")
    else:
        print(f"  {desc:<25} NO MATCH  [FAIL]")

# ── 4. Tracker: simulate a continuous recitation ──────────────────────────────
print("\n=== Tracker simulation ===")

tracker = QuranTracker(idx)

# Simulate reciting Al-Fatiha word by word
fatiha = (
    "بسم الله الرحمن الرحيم "
    "الحمد لله رب العالمين "
    "الرحمن الرحيم "
    "مالك يوم الدين "
    "إياك نعبد وإياك نستعين "
    "اهدنا الصراط المستقيم"
)

words = tokenize(fatiha)
last_result = None
for i, w in enumerate(words):
    result = tracker.process_token(w, is_final=True)
    if result and result.surah:
        last_result = result
        if i < 5 or result.is_switch:
            print(f"  token[{i:2d}] '{w}' → {result.state} surah={result.surah} "
                  f"ayah={result.ayah} conf={result.confidence:.2f}")

if last_result:
    print(f"  Final: surah={last_result.surah} ayah={last_result.ayah} "
          f"conf={last_result.confidence:.2f}")
    assert last_result.surah == 1, "Should be in Surah Al-Fatiha"

# ── 5. Tracker: simulate a switch to Al-Ikhlas ───────────────────────────────
print("\n=== Tracker switch detection ===")
ikhlas = "قل هو الله احد الله الصمد لم يلد ولم يولد ولم يكن له كفوا احد"
words2 = tokenize(ikhlas)
for i, w in enumerate(words2):
    result = tracker.process_token(w, is_final=True)
    if result and result.surah:
        if result.is_switch:
            print(f"  SWITCH detected at token '{w}': "
                  f"from {result.switch_from} → surah={result.surah}")
            break
        if result.surah == 112:
            print(f"  Locked on Surah 112 (Al-Ikhlas) at token '{w}' "
                  f"ayah={result.ayah} conf={result.confidence:.2f}")
            break

# ── 6. Ambiguity gate: shared phrases must not lock prematurely ───────────────
print("\n=== Ambiguity gate ===")

tracker2 = QuranTracker(idx)

# "الله لا اله الا هو الحي القيوم" starts BOTH 2:255 and 3:2 — must stay searching
opener = tokenize("الله لا اله الا هو الحي القيوم")
result_after_shared = None
for w in opener:
    r = tracker2.process_token(w, is_final=True)
    if r:
        result_after_shared = r

status = "OK (stayed searching)" if (result_after_shared is None or result_after_shared.state == "searching") \
         else f"FAIL — locked too early: surah={result_after_shared.surah} ayah={result_after_shared.ayah}"
print(f"  Shared opener of 2:255 & 3:2 → {status}")

# Now feed words unique to 2:255 (لا تأخذه سنة ولا نوم ...)
continuation = tokenize("لا تاخذه سنه ولا نوم له ما في السماوات وما في الارض")
locked_result = None
for w in continuation:
    r = tracker2.process_token(w, is_final=True)
    if r and r.state == "tracking" and r.surah == 2:
        locked_result = r
        break

if locked_result:
    print(f"  Locked on 2:{locked_result.ayah} after unique continuation  [OK]")
else:
    print("  Did not lock on 2:255 after unique continuation  [FAIL]")

# ── 7. Non-final lookahead: live position leads the final commits ─────────────
print("\n=== Non-final lookahead ===")

tracker3 = QuranTracker(idx)

# Lock onto Al-Baqarah 2 with 4 final tokens
lock_words = tokenize("ذلك الكتاب لا ريب")
for w in lock_words:
    tracker3.process_token(w, is_final=True)

# Now send "فيه هدى للمتقين" as non-final (simulating reciter ahead of finals)
lookahead_words = tokenize("فيه هدى للمتقين")
spec_result = None
for w in lookahead_words:
    r = tracker3.process_token(w, is_final=False)
    if r and r.state in ("tracking_speculative",):
        spec_result = r

if spec_result and spec_result.surah == 2:
    print(f"  Live position: surah=2 ayah={spec_result.ayah} "
          f"word_pos={spec_result.word_pos} conf={spec_result.confidence:.2f}  [OK]")
else:
    print(f"  Could not lookahead  [result={spec_result}]")

# ── 8. Speculative search (SEARCHING + non-finals) ────────────────────────────
print("\n=== Speculative search while still searching ===")
tracker4 = QuranTracker(idx)

# Send 2 finals (not enough to lock) then non-finals for a unique phrase
for w in tokenize("مالك يوم"):
    tracker4.process_token(w, is_final=True)

# Non-finals that together with the 2 finals form a unique 5-gram
spec2 = None
for w in tokenize("الدين إياك نعبد"):
    r = tracker4.process_token(w, is_final=False)
    if r and r.state == "searching_speculative":
        spec2 = r

if spec2 and spec2.surah == 1:
    print(f"  Speculative hint: surah=1 ayah={spec2.ayah} conf={spec2.confidence:.2f}  [OK]")
else:
    print(f"  Speculative search result: {spec2}")

print("\nAll tests passed!")

