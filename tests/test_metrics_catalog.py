"""Tests for metrics_catalog.py itself — guards against drift."""
import pytest

from core.metrics_catalog import (
    MetricKind,
    all_specs,
    by_layer,
    names,
    spec,
    validate_usage,
)


def test_catalog_has_no_duplicates():
    ns = names()
    assert len(ns) == len(set(ns))


def test_catalog_covers_every_layer():
    layers = set(by_layer().keys())
    for required in ("l4_strategy", "l5_execution", "l6_risk", "validation"):
        assert required in layers


def test_validate_usage_rejects_wrong_kind():
    with pytest.raises(AssertionError):
        validate_usage("account_balance", MetricKind.COUNTER)


def test_validate_usage_rejects_wrong_labels():
    with pytest.raises(AssertionError):
        validate_usage("risk_veto_emitted", MetricKind.COUNTER, labels=("symbol",))


def test_validate_usage_accepts_correct_labels():
    validate_usage("risk_veto_emitted", MetricKind.COUNTER,
                   labels=("symbol", "decision"))


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        spec("this_does_not_exist")
