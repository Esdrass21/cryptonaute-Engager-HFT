"""Tests for Metrics label handling (kubectl-style ``name{k=v}`` keys)."""
from core.metrics import Metrics, _metric_key


def test_inc_without_labels_uses_bare_name():
    m = Metrics()
    m.inc("ticks")
    m.inc("ticks")
    assert m.snapshot()["ticks"] == 2.0


def test_inc_accepts_label_kwargs():
    m = Metrics()
    m.inc("manheim_observation", source="manheim")
    assert m.snapshot()["manheim_observation{source=manheim}"] == 1.0


def test_inc_labels_are_sorted_and_merged():
    m = Metrics()
    m.inc("release_expected", source="ff", release_id="PPI_MOM")
    assert m.snapshot()["release_expected{release_id=PPI_MOM,source=ff}"] == 1.0


def test_inc_accepts_labels_dict_form():
    m = Metrics()
    m.inc("c", labels={"a": "1"})
    assert m.snapshot()["c{a=1}"] == 1.0


def test_inc_positional_value_still_works():
    m = Metrics()
    m.inc("amount", 2.5)
    assert m.snapshot()["amount"] == 2.5


def test_labeled_and_unlabeled_are_distinct():
    m = Metrics()
    m.inc("ff_feed_items", window="thisweek")
    m.inc("ff_feed_items", window="nextweek")
    m.inc("ff_feed_items")
    snap = m.snapshot()
    assert snap["ff_feed_items{window=thisweek}"] == 1.0
    assert snap["ff_feed_items{window=nextweek}"] == 1.0
    assert snap["ff_feed_items"] == 1.0


def test_metric_key_helper():
    assert _metric_key("x") == "x"
    assert _metric_key("x", {"b": "2", "a": "1"}) == "x{a=1,b=2}"
