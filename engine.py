"""Check engine: runs the pure checks in ``checks.py`` on a FiftyOne collection,
optionally tags the flagged labels, and returns a JSON-serializable report.

Used by the operators and from Python. No App/UI imports.

From Python, reach it through the operator, so no plugin module path is
needed::

    import fiftyone.operators as foo

    op = foo.get_operator("@tharanathrai/boxlint/check_labels")
    report = op.run_checks(view, "ground_truth", tag=False)
"""

from collections import Counter

import fiftyone as fo

from . import checks as _checks

ISSUE_PREFIX = "issue:"

ERROR = "error"
WARNING = "warning"

DETECTIONS = "detections"
POLYLINES = "polylines"

# check_id -> (field kind, severity, label, description)
CHECKS = {
    "degenerate": (
        DETECTIONS,
        ERROR,
        "Degenerate boxes",
        "Width or height <= 0, or any NaN/inf coordinate",
    ),
    "out_of_bounds": (
        DETECTIONS,
        ERROR,
        "Out-of-bounds boxes",
        "Any edge outside the image by more than the tolerance",
    ),
    "tiny": (
        DETECTIONS,
        WARNING,
        "Tiny boxes",
        "Pixel area below the minimum (needs image size)",
    ),
    "aspect_ratio": (
        DETECTIONS,
        WARNING,
        "Extreme aspect ratio",
        "Pixel width / height outside [1/max, max] (needs image size)",
    ),
    "duplicate": (
        DETECTIONS,
        WARNING,
        "Duplicate boxes",
        "Same-class boxes on one image overlapping above the IoU threshold; "
        "the lower-confidence (or later) one is tagged",
    ),
    "unknown_class": (
        DETECTIONS,
        ERROR,
        "Unknown classes",
        "Class label not in the allowed list (exact, case-sensitive)",
    ),
    "bad_polygon": (
        POLYLINES,
        ERROR,
        "Bad polygons",
        "Fewer than 3 points, zero area, or self-intersecting. Open "
        "polylines (not closed, not filled) are skipped",
    ),
}

# Checks that need sample.metadata.width/height
PIXEL_CHECKS = ("tiny", "aspect_ratio")

# Runs only when an allowed-class list is given
VOCAB_CHECKS = ("unknown_class",)

DEFAULTS = {
    "oob_tol": 0.001,
    "min_area_px": 16,
    "max_ratio": 20,
    "iou_thresh": 0.95,
}


def run_checks(
    collection,
    label_field,
    checks=None,
    oob_tol=DEFAULTS["oob_tol"],
    min_area_px=DEFAULTS["min_area_px"],
    max_ratio=DEFAULTS["max_ratio"],
    iou_thresh=DEFAULTS["iou_thresh"],
    allowed_classes=None,
    compute_metadata=True,
    clear_previous=True,
    tag=True,
):
    """Runs label checks on one Detections or Polylines field.

    Args:
        collection: a FiftyOne dataset or view
        label_field: name of a Detections or Polylines field
        checks (None): iterable of check IDs to run. By default, every check
            that applies to the field's type, with ``unknown_class`` only when
            ``allowed_classes`` is given
        oob_tol, min_area_px, max_ratio, iou_thresh: check thresholds
        allowed_classes (None): list of class names, or a comma-separated
            string. Enables ``unknown_class``
        compute_metadata (True): compute missing image width/height for the
            pixel checks. This writes ``sample.metadata``, never labels. If
            False, those samples are skipped and the report says so
        clear_previous (True): remove existing ``issue:*`` tags from the field
            in ``collection`` before tagging. Ignored when ``tag`` is False
        tag (True): tag flagged labels ``issue:<check_id>``. False is a dry
            run that writes no tags

    Returns:
        a JSON-serializable report dict; see ``SPEC.md``

    Raises:
        ValueError: on an unknown field, check ID, or invalid threshold. All
            inputs are validated before anything is written
    """
    field_kinds = get_label_fields(collection)
    kind = field_kinds.get(label_field)
    if kind is None:
        raise ValueError(
            "%r is not a Detections or Polylines field; available: %s"
            % (label_field, sorted(field_kinds) or "none")
        )

    allowed = parse_allowed_classes(allowed_classes)
    enabled = _resolve_enabled(checks, kind, allowed)

    thresholds = {
        "oob_tol": oob_tol,
        "min_area_px": min_area_px,
        "max_ratio": max_ratio,
        "iou_thresh": iou_thresh,
    }
    _validate_thresholds(thresholds)

    skipped = []
    pixel_enabled = [c for c in enabled if c in PIXEL_CHECKS]
    if pixel_enabled and compute_metadata and count_missing_metadata(collection):
        collection.compute_metadata()

    if kind == DETECTIONS:
        labels, sizes = _get_detections(collection, label_field)
        results, extra = _run_detection_checks(
            labels, sizes, enabled, thresholds, allowed, skipped
        )
    else:
        labels = _get_polylines(collection, label_field)
        results, extra = _run_polyline_checks(labels, enabled, skipped)

    cleared = 0
    if tag:
        if clear_previous:
            cleared = clear_issue_tags(collection, label_field)

        for check_id, label_ids in results.items():
            if label_ids:
                collection.select_labels(
                    ids=label_ids, fields=label_field
                ).tag_labels(ISSUE_PREFIX + check_id, label_fields=label_field)

    return _build_report(
        collection,
        label_field,
        kind,
        labels,
        results,
        extra,
        skipped,
        cleared,
        config={
            "checks": enabled,
            **thresholds,
            "allowed_classes": allowed,
            "compute_metadata": bool(compute_metadata),
            "clear_previous": bool(clear_previous),
            "tag": bool(tag),
        },
    )


def parse_allowed_classes(allowed_classes):
    """Returns a list of class names from a list or a comma-separated string."""
    if isinstance(allowed_classes, str):
        return _checks.parse_class_list(allowed_classes)

    return [c for c in (allowed_classes or []) if c is not None]


def get_label_fields(collection):
    """Returns ``{field_name: "detections" | "polylines"}``."""
    if collection is None:
        return {}

    kinds = {}
    for doc_type, kind in ((fo.Detections, DETECTIONS), (fo.Polylines, POLYLINES)):
        for name in collection.get_field_schema(embedded_doc_type=doc_type):
            kinds[name] = kind

    return kinds


def declared_classes(dataset, field):
    """Returns the dataset's declared classes for ``field``, else its default
    classes, else an empty list.
    """
    if dataset is None:
        return []

    return list((dataset.classes or {}).get(field) or dataset.default_classes or [])


def count_missing_metadata(collection):
    return collection.exists("metadata.width", False).count()


def clear_issue_tags(collection, label_fields):
    """Removes all ``issue:*`` tags from the given field(s) of ``collection``.

    Returns:
        ``{tag: count}`` of the tags removed
    """
    counts = collection.count_label_tags(label_fields=label_fields)
    removed = {t: n for t, n in counts.items() if t.startswith(ISSUE_PREFIX)}
    if removed:
        collection.untag_labels(list(removed), label_fields=label_fields)

    return removed


def applicable_checks(kind):
    return [c for c, spec in CHECKS.items() if spec[0] == kind]


def _resolve_enabled(requested, kind, allowed):
    applicable = applicable_checks(kind)
    if requested is None:
        return [
            c for c in applicable if c not in VOCAB_CHECKS or allowed
        ]

    requested = list(requested)
    unknown = [c for c in requested if c not in CHECKS]
    if unknown:
        raise ValueError(
            "Unknown check(s) %s; available: %s" % (unknown, sorted(CHECKS))
        )

    wrong_kind = [c for c in requested if c not in applicable]
    if wrong_kind:
        raise ValueError(
            "Check(s) %s do not apply to a %s field" % (wrong_kind, kind)
        )

    if any(c in VOCAB_CHECKS for c in requested) and not allowed:
        raise ValueError("unknown_class needs a non-empty allowed_classes")

    # Registry order, no repeats
    return [c for c in applicable if c in requested]


def _validate_thresholds(t):
    # The pure checks validate their own thresholds; call them on no data so
    # bad values fail before anything is written
    _checks.find_out_of_bounds([], tol=t["oob_tol"])
    _checks.find_tiny([], min_area_px=t["min_area_px"])
    _checks.find_extreme_aspect_ratio([], max_ratio=t["max_ratio"])
    _checks.find_duplicates([], iou_thresh=t["iou_thresh"])


def _get_detections(collection, field):
    """Returns ``[{id, label, box, confidence, sample_id}]`` and
    ``{sample_id: (width, height) | None}``.
    """
    prefix = field + ".detections."
    sample_ids, widths, heights, ids, labels, boxes, confs = collection.values(
        [
            "id",
            "metadata.width",
            "metadata.height",
            prefix + "id",
            prefix + "label",
            prefix + "bounding_box",
            prefix + "confidence",
        ]
    )

    out = []
    sizes = {}
    for sample_id, w, h, s_ids, s_labels, s_boxes, s_confs in zip(
        sample_ids, widths, heights, ids, labels, boxes, confs
    ):
        sizes[sample_id] = (w, h) if w is not None and h is not None else None
        s_ids, s_labels, s_boxes, s_confs = _aligned(
            field, s_ids, s_labels, s_boxes, s_confs
        )
        for label_id, label, box, conf in zip(s_ids, s_labels, s_boxes, s_confs):
            out.append(
                {
                    "id": label_id,
                    "label": label,
                    "box": box,
                    "confidence": conf,
                    "sample_id": sample_id,
                }
            )

    return out, sizes


def _get_polylines(collection, field):
    """Returns ``[{id, label, points, closed, filled, sample_id}]``."""
    prefix = field + ".polylines."
    sample_ids, ids, labels, points, closed, filled = collection.values(
        [
            "id",
            prefix + "id",
            prefix + "label",
            prefix + "points",
            prefix + "closed",
            prefix + "filled",
        ]
    )

    out = []
    for sample_id, s_ids, s_labels, s_points, s_closed, s_filled in zip(
        sample_ids, ids, labels, points, closed, filled
    ):
        s_ids, s_labels, s_points, s_closed, s_filled = _aligned(
            field, s_ids, s_labels, s_points, s_closed, s_filled
        )
        for label_id, label, pts, c, f in zip(
            s_ids, s_labels, s_points, s_closed, s_filled
        ):
            out.append(
                {
                    "id": label_id,
                    "label": label,
                    "points": pts,
                    "closed": c,
                    "filled": f,
                    "sample_id": sample_id,
                }
            )

    return out


def _aligned(field, ids, *columns):
    """Returns ``ids`` and ``columns`` as lists (None, for a sample without the
    field, becomes []), raising if any column length differs from ``ids``,
    since zip() would silently drop labels.
    """
    ids = ids or []
    columns = [col or [] for col in columns]
    for col in columns:
        if len(col) != len(ids):
            raise RuntimeError(
                "Label attributes of %r are misaligned (%d ids vs %d values)"
                % (field, len(ids), len(col))
            )

    return [ids] + columns


def _run_detection_checks(labels, sizes, enabled, t, allowed, skipped):
    boxes = [(d["id"], d["box"]) for d in labels]
    pixel_boxes = [(d["id"], d["box"], sizes[d["sample_id"]]) for d in labels]

    results = {}
    if "degenerate" in enabled:
        results["degenerate"] = _checks.find_degenerate(boxes)

    if "out_of_bounds" in enabled:
        results["out_of_bounds"] = _checks.find_out_of_bounds(
            boxes, tol=t["oob_tol"]
        )

    pixel_enabled = [c for c in enabled if c in PIXEL_CHECKS]
    if pixel_enabled:
        no_size = {s for s, size in sizes.items() if size is None}
        if no_size:
            num_labels = sum(1 for d in labels if d["sample_id"] in no_size)
            skipped.append(
                "%s: skipped %d labels on %d samples with no image size"
                % (", ".join(pixel_enabled), num_labels, len(no_size))
            )

    if "tiny" in enabled:
        results["tiny"] = _checks.find_tiny(
            pixel_boxes, min_area_px=t["min_area_px"]
        )

    if "aspect_ratio" in enabled:
        results["aspect_ratio"] = _checks.find_extreme_aspect_ratio(
            pixel_boxes, max_ratio=t["max_ratio"]
        )

    if "duplicate" in enabled:
        by_sample = {}
        for d in labels:
            by_sample.setdefault(d["sample_id"], []).append(
                (d["id"], d["label"], d["box"], d["confidence"])
            )
        results["duplicate"] = [
            label_id
            for sample_labels in by_sample.values()
            for label_id in _checks.find_duplicates(
                sample_labels, iou_thresh=t["iou_thresh"]
            )
        ]

    if "unknown_class" in enabled:
        results["unknown_class"] = _checks.find_unknown_class(
            [(d["id"], d["label"]) for d in labels], allowed
        )

    return results, {}


def _run_polyline_checks(labels, enabled, skipped):
    results = {}
    extra = {}
    if "bad_polygon" in enabled:
        rows = [(p["id"], p["points"], p["closed"], p["filled"]) for p in labels]
        try:
            flagged = _checks.find_bad_polygons(rows)
        except ImportError:
            skipped.append("bad_polygon: skipped, shapely is not installed")
        else:
            results["bad_polygon"] = flagged
            points = {p["id"]: p["points"] for p in labels}
            reasons = Counter()
            for label_id in flagged:
                shapes = points[label_id] or []
                problems = [_checks.polygon_problem(s) for s in shapes]
                reasons[next((r for r in problems if r), "no_shapes")] += 1
            extra["bad_polygon"] = {"reason_counts": dict(reasons.most_common())}

    return results, extra


def _build_report(
    collection, field, kind, labels, results, extra, skipped, cleared, config
):
    label_class = {d["id"]: d["label"] for d in labels}
    label_sample = {d["id"]: d["sample_id"] for d in labels}

    report_checks = {}
    for check_id, label_ids in results.items():
        classes = Counter(str(label_class[i]) for i in label_ids)
        report_checks[check_id] = {
            "severity": CHECKS[check_id][1],
            "tag": ISSUE_PREFIX + check_id,
            "num_labels": len(label_ids),
            "num_samples": len({label_sample[i] for i in label_ids}),
            "class_counts": dict(classes.most_common()),
            "top_classes": [c for c, _ in classes.most_common(3)],
            "label_ids": list(label_ids),
            **extra.get(check_id, {}),
        }

    def _distinct(severity=None):
        return {
            i
            for c, ids in results.items()
            if severity is None or CHECKS[c][1] == severity
            for i in ids
        }

    flagged = _distinct()
    return {
        "label_field": field,
        "field_type": kind,
        "num_samples": collection.count(),
        "num_labels_checked": len(labels),
        "num_flagged_labels": len(flagged),
        "num_flagged_samples": len({label_sample[i] for i in flagged}),
        "num_error_labels": len(_distinct(ERROR)),
        "num_warning_labels": len(_distinct(WARNING)),
        "num_tags_cleared": sum(cleared.values()) if cleared else 0,
        "checks": report_checks,
        "skipped": skipped,
        "config": config,
    }
