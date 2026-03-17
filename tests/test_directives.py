"""Tests for the directives module."""

from autorefine.directives import DirectiveManager, RefinementDirectives
from autorefine.storage.json_store import JSONStore


def _make_store(tmp_path):
    return JSONStore(str(tmp_path / "test_store.json"))


def test_set_directives(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    rd = mgr.set(
        "default",
        directives=["Be concise", "Cite sources"],
        domain_context="A knowledge bot",
        preserve_behaviors=["Keep bullet points"],
    )
    assert rd.version == 1
    assert len(rd.directives) == 2
    assert rd.domain_context == "A knowledge bot"


def test_get_directives(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    mgr.set("default", directives=["Rule 1"])
    rd = mgr.get("default")
    assert rd is not None
    assert rd.directives == ["Rule 1"]


def test_get_nonexistent(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)
    assert mgr.get("nonexistent") is None


def test_set_increments_version(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    mgr.set("default", directives=["v1"])
    rd2 = mgr.set("default", directives=["v2"])
    assert rd2.version == 2
    assert rd2.directives == ["v2"]


def test_update_adds_directives(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    mgr.set("default", directives=["Rule A"])
    rd = mgr.update("default", add_directives=["Rule B"])
    assert "Rule A" in rd.directives
    assert "Rule B" in rd.directives


def test_update_removes_directives(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    mgr.set("default", directives=["Rule A", "Rule B", "Rule C"])
    rd = mgr.update("default", remove_directives=["Rule B"])
    assert "Rule B" not in rd.directives
    assert "Rule A" in rd.directives
    assert "Rule C" in rd.directives


def test_format_for_meta_prompt_empty(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)
    assert mgr.format_for_meta_prompt("default") == ""


def test_format_for_meta_prompt(tmp_path):
    store = _make_store(tmp_path)
    mgr = DirectiveManager(store)

    mgr.set(
        "default",
        directives=["Never mention competitors", "Always cite sources"],
        domain_context="Customer support bot for SaaS product",
        preserve_behaviors=["The friendly greeting"],
    )
    text = mgr.format_for_meta_prompt("default")

    assert "NON-NEGOTIABLE" in text
    assert "DOMAIN CONTEXT" in text
    assert "Customer support bot" in text
    assert "HARD CONSTRAINTS" in text
    assert "Never mention competitors" in text
    assert "BEHAVIORS TO PRESERVE" in text
    assert "friendly greeting" in text
