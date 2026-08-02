import glob
import os
import shutil
import struct
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()

# Compaction (whole-snapshot mode, --slot-save-incremental OFF): after a successful whole-root
# auto-cache save, shorter exact-prefix auto-*.bin snapshots are deleted to reclaim disk. Manual
# saves, pinned snapshots and snapshots that still parent a live delta are never compacted.

CACHE_DIR = "./tmp/slot_save_compact"

SLOT_META_MAGIC = 0x544D4B4C  # "LKMT", LE
SLOT_META_TOKS_OFF = 104      # offset of the trailing-token-count u32 in the header
SLOT_META_VERSION_NODE = 3


def _join(sentence: str, n: int) -> str:
    return " ".join([sentence] * n)


# BASE is long enough to clear the small hash block; TAIL strict-extends it without merging
# tokens (BASE has no trailing space, TAIL begins with one) so BASE's ids stay an exact prefix.
BASE = _join("Once upon a time there was a little dog named Spot who loved to run.", 10)
TAIL = " " + _join("The dog ran fast across the green field every single morning.", 3)
P1 = BASE
P2 = BASE + TAIL


def parse_meta(path: str):
    """Parse a .meta sidecar (v1 whole / v3 delta) — returns version, tok_count, chain_hash."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == SLOT_META_MAGIC, f"bad magic in {path}"
    tok_count = struct.unpack_from("<I", data, SLOT_META_TOKS_OFF)[0]
    chain_hash = struct.unpack_from("<Q", data, SLOT_META_TOKS_OFF + 4)[0]
    return {"version": version, "tok_count": tok_count, "chain_hash": chain_hash}


def _metas():
    return sorted(glob.glob(os.path.join(CACHE_DIR, "auto-*.meta")))


def _bin_for(meta_path: str) -> str:
    assert meta_path.endswith(".meta")
    return meta_path[: -len(".meta")]


def _roots(metas):
    return [m for m in metas if parse_meta(m)["version"] == 1]


def _deltas(metas):
    return [m for m in metas if parse_meta(m)["version"] == SLOT_META_VERSION_NODE]


@pytest.fixture(autouse=True)
def clean_cache_dir():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR)
    yield
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _mk(incremental: bool, compact: bool = True, n_ctx: int = 512, n_slots: int = 1):
    s = ServerPreset.tinyllama2()
    s.n_ctx = n_ctx
    s.n_batch = 512
    s.n_slots = n_slots
    s.temperature = 0.0
    s.seed = 42
    s.slot_save_path = CACHE_DIR
    s.slot_save_auto = True
    s.slot_save_incremental = incremental
    s.slot_save_compact = compact
    s.slot_save_block = 16        # small block so short test prompts clear the boundary
    s.slot_save_min_tokens = 0    # keep the floor at the hash block size
    s.slot_save_idle_seconds = -1  # rely on the deterministic shutdown flush, not the idle timer
    return s


def _complete(s, prompt, n_predict=4, id_slot=0):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": id_slot,
    })
    assert res.status_code == 200
    return res.body


# --- (1) whole-snapshot mode compacts a shorter exact-prefix snapshot ------------

def test_compact_deletes_shorter_prefix_whole_snapshot():
    """With --slot-save-incremental OFF, saving a longer whole snapshot that strict-extends an
    existing shorter one deletes the shorter one (and its sidecars): a growing conversation must
    not leave a trail of obsolete partial snapshots on disk."""
    global server

    # session 1: persist P1 as a whole root (shutdown flush).
    server = _mk(incremental=False)
    server.start()
    _complete(server, P1, n_predict=0)
    server.stop()
    m1 = _metas()
    assert len(m1) == 1, "P1 must be persisted as one whole snapshot"
    p1_meta = parse_meta(m1[0])
    assert p1_meta["version"] == 1

    # session 2 (fresh process, shared dir): restore P1 from disk, extend to P2, then the
    # shutdown flush publishes P2 as a whole root and compaction deletes the P1 prefix.
    server = _mk(incremental=False)
    server.start()
    _complete(server, P2, n_predict=4)
    server.stop()

    m2 = _metas()
    assert len(m2) == 1, "compaction must delete the shorter P1 prefix, leaving only P2"
    p2_meta = parse_meta(m2[0])
    assert p2_meta["version"] == 1
    assert p2_meta["tok_count"] > p1_meta["tok_count"], "the survivor must be the longer P2 snapshot"
    # the deleted P1 .bin / .logits sidecars are gone too
    assert not os.path.exists(_bin_for(m1[0])), "the compacted P1 .bin must be removed"
    assert not os.path.exists(_bin_for(m1[0]) + ".logits"), "the compacted P1 .logits must be removed"


# --- (2) --no-slot-save-compact preserves shorter prefixes ----------------------

def test_no_compact_preserves_shorter_prefix():
    """--no-slot-save-compact disables compaction: both the shorter and the longer whole
    snapshots survive on disk."""
    global server

    server = _mk(incremental=False, compact=False)
    server.start()
    _complete(server, P1, n_predict=0)
    server.stop()
    assert len(_metas()) == 1

    server = _mk(incremental=False, compact=False)
    server.start()
    _complete(server, P2, n_predict=4)
    server.stop()

    metas = _metas()
    assert len(metas) == 2, "with compaction disabled both whole snapshots must survive"
    counts = sorted(parse_meta(m)["tok_count"] for m in metas)
    assert counts[0] < counts[1], "the two survivors are the shorter P1 and the longer P2"


# --- (3) manual saves are never compacted ---------------------------------------

def test_manual_save_survives_compaction():
    """A manual /slots save (a non-auto-* filename) is never compacted, even when its tokens are
    an exact prefix of a later auto-cache whole snapshot. Only auto-* files are compaction
    candidates."""
    global server

    # session 1: persist P1 both as a manual save AND as the shutdown-flush auto snapshot.
    server = _mk(incremental=False)
    server.server_slots = True  # expose the manual /slots endpoints
    server.start()
    _complete(server, P1, n_predict=0)
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "keep.bin"})
    assert res.status_code == 200
    server.stop()
    assert os.path.exists(os.path.join(CACHE_DIR, "keep.bin"))
    assert os.path.exists(os.path.join(CACHE_DIR, "keep.bin.meta"))
    assert len(_metas()) == 1  # the auto P1 snapshot

    # session 2: extend to P2; the auto P2 save compacts the auto P1 but spares the manual save.
    server = _mk(incremental=False)
    server.start()
    _complete(server, P2, n_predict=4)
    server.stop()

    # the manual save (a prefix of P2) survives compaction
    assert os.path.exists(os.path.join(CACHE_DIR, "keep.bin")), "a manual save must never be compacted"
    assert os.path.exists(os.path.join(CACHE_DIR, "keep.bin.meta")), \
        "a manual save's .meta sidecar must never be compacted"
    # exactly one auto snapshot remains (P2); the auto P1 was compacted
    metas = _metas()
    assert len(metas) == 1, "the auto P1 prefix must be compacted, leaving the auto P2"
    assert parse_meta(metas[0])["tok_count"] > parse_meta(os.path.join(CACHE_DIR, "keep.bin.meta"))["tok_count"]


# --- (4) incremental mode never compacts (deltas + roots coexist) ----------------

def test_incremental_mode_does_not_compact():
    """With --slot-save-incremental ON, compaction is a no-op even though P2 strict-extends P1:
    the base root and the delta node both survive (delta trees manage their own space via LRU
    and parent/child refcounting, not prefix compaction)."""
    global server

    server = _mk(incremental=True)
    server.start()
    _complete(server, P1, n_predict=0)
    server.stop()
    assert len(_metas()) == 1  # P1 root

    server = _mk(incremental=True)
    server.start()
    _complete(server, P2, n_predict=4)
    server.stop()

    metas = _metas()
    assert len(metas) == 2, "incremental mode must keep the root + the delta (no compaction)"
    assert len(_roots(metas)) == 1 and len(_deltas(metas)) == 1, \
        "expected one v1 root + one v3 delta, both preserved"


# --- (5) incremental caches survive a non-incremental restart (deltas never compacted) ----

def test_delta_nodes_and_their_parents_survive_non_incremental_restart():
    """Compaction only ever deletes WHOLE-ROOT snapshots, never delta nodes, and a whole root that
    still parents a live delta is protected. So a directory holding an incremental tree (root +
    delta) written to by a server restarted WITHOUT --slot-save-incremental keeps BOTH the root
    and the delta: the delta is never a compaction candidate, and the root is shielded by its live
    delta child. Only the new whole snapshot is added. (The delta's space is later reclaimed by the
    tree-aware LRU, never by prefix compaction.)

    Once the delta is gone (a peer/LRU eviction), a further whole save DOES compact the now-
    childless root: only whole roots with no live delta children and an exact-prefix relationship
    are ever compacted."""
    global server

    # a prompt family where every member is a stable exact prefix of the next (each ends in "run."
    # and the next repetition begins with " Once", the proven-stable ". " boundary).
    s1 = "Once upon a time there was a little dog named Spot who loved to run."
    p1 = _join(s1, 5)
    p2 = _join(s1, 10)   # p1 is an exact prefix of p2
    p3 = _join(s1, 15)   # p2 is an exact prefix of p3
    p4 = _join(s1, 20)   # p3 is an exact prefix of p4

    # build an incremental tree: p1 root + p2 delta (n_predict=0 => snapshots are pure prompts).
    server = _mk(incremental=True)
    server.start()
    _complete(server, p1, n_predict=0)
    server.stop()
    server = _mk(incremental=True)
    server.start()
    _complete(server, p2, n_predict=0)
    server.stop()
    metas = _metas()
    assert len(metas) == 2 and len(_deltas(metas)) == 1, "need a root + delta before the switch"
    delta_path = _deltas(metas)[0]
    root_path = _roots(metas)[0]
    root_len = parse_meta(root_path)["tok_count"]

    # restart WITHOUT incremental and save a longer whole snapshot (p3) that strict-extends p2.
    server = _mk(incremental=False)
    server.start()
    _complete(server, p3, n_predict=0)
    server.stop()

    # the delta node is NEVER compacted (only whole roots are candidates)
    assert os.path.exists(_bin_for(delta_path)), \
        "a delta node must never be compacted, even when it is an exact prefix of a new whole save"
    assert os.path.exists(delta_path), "the delta .meta must survive"
    # the root is protected this pass because its delta child is still on disk
    assert os.path.exists(_bin_for(root_path)), \
        "a whole root with a live delta child must survive compaction (no orphan delta left behind)"
    # the new whole p3 snapshot was added -> root + delta + p3 whole
    metas_after = _metas()
    assert len(metas_after) == 3, \
        f"root + delta + new whole must all survive, got {len(metas_after)} metas"
    p3_len = max(parse_meta(m)["tok_count"] for m in metas_after)
    assert p3_len > parse_meta(delta_path)["tok_count"], \
        "the longest survivor must be the new whole p3 snapshot"

    # simulate the delta being evicted (by a peer / the tree-aware LRU), then a further whole
    # save compacts the now-childless root (and the earlier p3 whole): only whole roots with no
    # live delta children and an exact-prefix relationship are compacted.
    for suffix in ("", ".meta", ".logits"):
        victim = _bin_for(delta_path) + suffix
        if os.path.exists(victim):
            os.remove(victim)
    server = _mk(incremental=False)
    server.start()
    _complete(server, p4, n_predict=0)
    server.stop()
    final = _metas()
    final_counts = {parse_meta(m)["tok_count"] for m in final}
    assert root_len not in final_counts, \
        "the now-childless root must be compacted once its delta child is gone"
    assert p3_len not in final_counts, \
        "the earlier p3 whole (an exact prefix of p4) must be compacted too"
    # only the new longest whole p4 snapshot remains
    assert len(final) == 1, f"only the p4 whole should remain, got {len(final)} metas"
    assert max(final_counts) > p3_len, "the survivor must be the longer p4 whole snapshot"
