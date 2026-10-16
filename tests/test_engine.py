"""Tests for engine.run_checks on a small in-memory dataset."""

import json
import uuid

import pytest

fo = pytest.importorskip("fiftyone")

from conftest import load_plugin_package

engine = load_plugin_package().engine

BOX = [0.1, 0.1, 0.3, 0.3]
SQUARE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
BOWTIE = [(0.0, 0.0), (0.4, 0.4), (0.4, 0.0), (0.0, 0.4)]


def _det(label, box=BOX, confidence=None):
    return fo.Detection(label=label, bounding_box=box, confidence=confidence)


@pytest.fixture
def dataset():
    ds = fo.Dataset("lsc-test-" + uuid.uuid4().hex[:8])
    meta = fo.ImageMetadata(width=100, height=100)
    s1 = fo.Sample(
        filepath="/tmp/lsc-1.jpg",
        metadata=meta,
        gt=fo.Detections(
            detections=[
                _det("cat"),
                _det("cat"),  # duplicate (warning)
                _det("cat", [0.2, 0.2, 0.0, 0.3]),  # degenerate (error)
                _det("dog", [0.9, 0.1, 0.3, 0.3]),  # out of bounds (error)
                _det("Dog"),  # unknown class when allowed = cat, dog
            ]
        ),
        polys=fo.Polylines(
            polylines=[
                fo.Polyline(label="a", points=[SQUARE], closed=True, filled=True),
                fo.Polyline(label="b", points=[BOWTIE], closed=True, filled=True),
                fo.Polyline(label="c", points=[], closed=True, filled=True),
                fo.Polyline(label="d", points=[BOWTIE], closed=False, filled=False),
            ]
        ),
    )
    s2 = fo.Sample(
        filepath="/tmp/lsc-2.jpg",
        metadata=meta,
        gt=fo.Detections(
            detections=[_det("cat", [0.1, 0.1, 0.02, 0.02])]  # tiny: 4 px area
        ),
    )
    s3 = fo.Sample(filepath="/tmp/lsc-3.jpg", gt=fo.Detections(detections=[_det("cat")]))
    ds.add_samples([s1, s2, s3])
    yield ds
    ds.delete()


def _tags(ds, field="gt"):
    return ds.count_label_tags(label_fields=field)


def test_report_shape_and_json(dataset):
    report = engine.run_checks(dataset, "gt", compute_metadata=False)
    json.dumps(report)  # serializable

    assert set(report) == {
        "label_field", "field_type", "num_samples", "num_labels_checked",
        "num_flagged_labels", "num_flagged_samples", "num_error_labels",
        "num_warning_labels", "num_tags_cleared", "checks", "skipped", "config",
    }
    assert report["field_type"] == "detections"
    assert report["num_samples"] == 3
    assert report["num_labels_checked"] == 7

    for check_id, result in report["checks"].items():
        assert result["tag"] == "issue:" + check_id
        assert result["severity"] in ("error", "warning")
        assert result["num_labels"] == len(result["label_ids"])
        assert len(result["top_classes"]) <= 3


def test_default_checks_and_counts(dataset):
    report = engine.run_checks(dataset, "gt", compute_metadata=False)
    counts = {c: r["num_labels"] for c, r in report["checks"].items()}
    assert counts == {
        "degenerate": 1,
        "out_of_bounds": 1,
        "tiny": 1,
        "aspect_ratio": 0,
        "duplicate": 1,
    }
    assert "unknown_class" not in report["checks"]  # no vocabulary given
    assert report["num_error_labels"] == 2
    assert report["num_warning_labels"] == 2
    assert report["num_flagged_labels"] == 4
    assert report["num_flagged_samples"] == 2
    assert report["checks"]["duplicate"]["class_counts"] == {"cat": 1}


def test_unknown_class_enabled_by_allowed_list(dataset):
    report = engine.run_checks(dataset, "gt", allowed_classes="cat, dog", compute_metadata=False)
    result = report["checks"]["unknown_class"]
    assert result["num_labels"] == 1
    assert result["class_counts"] == {"Dog": 1}
    assert result["severity"] == "error"
    assert report["config"]["allowed_classes"] == ["cat", "dog"]

    as_list = engine.run_checks(dataset, "gt", allowed_classes=["cat", "dog"], compute_metadata=False)
    assert as_list["checks"]["unknown_class"]["label_ids"] == result["label_ids"]


def test_tags_written_and_idempotent(dataset):
    engine.run_checks(dataset, "gt", compute_metadata=False)
    first = _tags(dataset)
    report = engine.run_checks(dataset, "gt", compute_metadata=False)
    assert _tags(dataset) == first
    assert report["num_tags_cleared"] == sum(first.values())


def test_dry_run_writes_nothing(dataset):
    dataset.select_labels(fields="gt").limit(1).tag_labels("issue:old", label_fields="gt")
    before = _tags(dataset)
    report = engine.run_checks(dataset, "gt", tag=False, compute_metadata=False)
    assert _tags(dataset) == before
    assert report["num_tags_cleared"] == 0
    assert report["checks"]["degenerate"]["num_labels"] == 1


def test_clear_previous_false_keeps_old_tags(dataset):
    dataset.tag_labels("issue:old", label_fields="gt")
    engine.run_checks(dataset, "gt", clear_previous=False, compute_metadata=False)
    assert "issue:old" in _tags(dataset)

    engine.run_checks(dataset, "gt", clear_previous=True, compute_metadata=False)
    assert "issue:old" not in _tags(dataset)


def test_non_issue_tags_untouched(dataset):
    dataset.tag_labels("reviewed", label_fields="gt")
    engine.run_checks(dataset, "gt", compute_metadata=False)
    assert _tags(dataset)["reviewed"] == 7


def test_missing_metadata_reported_not_silent(dataset):
    report = engine.run_checks(dataset, "gt", compute_metadata=False)
    assert report["skipped"] == [
        "tiny, aspect_ratio: skipped 1 labels on 1 samples with no image size"
    ]


def test_explicit_checks_subset(dataset):
    report = engine.run_checks(dataset, "gt", checks=["degenerate"])
    assert list(report["checks"]) == ["degenerate"]
    assert report["skipped"] == []  # pixel checks not run, nothing to skip
    assert set(_tags(dataset)) == {"issue:degenerate"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_field": "nope"},
        {"label_field": "gt", "checks": ["bogus"]},
        {"label_field": "gt", "checks": ["bad_polygon"]},
        {"label_field": "polys", "checks": ["degenerate"]},
        {"label_field": "gt", "checks": ["unknown_class"]},
        {"label_field": "gt", "min_area_px": -1},
        {"label_field": "gt", "iou_thresh": 1.5},
        {"label_field": "gt", "max_ratio": 0.5},
        {"label_field": "gt", "oob_tol": float("nan")},
    ],
)
def test_invalid_input_raises_before_writing(dataset, kwargs):
    dataset.tag_labels("issue:old", label_fields="gt")
    field = kwargs.pop("label_field")
    with pytest.raises(ValueError):
        engine.run_checks(dataset, field, **kwargs)
    assert _tags(dataset) == {"issue:old": 7}


def test_polylines_report(dataset):
    report = engine.run_checks(dataset, "polys")
    assert report["field_type"] == "polylines"
    assert report["num_labels_checked"] == 4
    result = report["checks"]["bad_polygon"]
    assert result["num_labels"] == 2  # bow-tie and empty; open line skipped
    assert result["reason_counts"] == {"self_intersecting": 1, "no_shapes": 1}
    assert result["class_counts"] == {"b": 1, "c": 1}
    assert _tags(dataset, "polys") == {"issue:bad_polygon": 2}
    assert _tags(dataset, "gt") == {}  # other field untouched


def test_view_scoping(dataset):
    view = dataset.limit(1)
    report = engine.run_checks(view, "gt", compute_metadata=False)
    assert report["num_samples"] == 1
    assert "tiny" in report["checks"] and report["checks"]["tiny"]["num_labels"] == 0


def test_clear_issue_tags_returns_counts(dataset):
    engine.run_checks(dataset, "gt", compute_metadata=False)
    engine.run_checks(dataset, "polys")
    dataset.tag_labels("reviewed", label_fields="gt")

    removed = engine.clear_issue_tags(dataset, ["gt", "polys"])
    assert removed == {
        "issue:degenerate": 1,
        "issue:out_of_bounds": 1,
        "issue:duplicate": 1,
        "issue:tiny": 1,
        "issue:bad_polygon": 2,
    }
    assert _tags(dataset, "gt") == {"reviewed": 7}
    assert _tags(dataset, "polys") == {}


def test_operator_exposes_python_api():
    pkg = load_plugin_package()
    assert pkg.CheckLabels.run_checks is engine.run_checks
    assert pkg.ClearIssueTags.clear_issue_tags is engine.clear_issue_tags
