"""Pure check functions for boxlint.

No FiftyOne imports here, so every check is unit-testable in isolation.

Box checks take an iterable of ``(label_id, bounding_box)`` pairs, where
``bounding_box`` is FiftyOne's ``[x, y, w, h]`` in relative coordinates with a
top-left origin. Pixel checks take ``(label_id, bounding_box, (width, height))``
triples, where the image size is in pixels. Each check returns the list of
flagged label IDs, in input order.

Polygon checks need ``shapely``, imported lazily so the box checks work
without it.
"""

import math

# Slack added to boundary comparisons so float rounding (e.g. 0.3 + 0.7) never
# produces a flag. Conservative: a false positive costs more than a miss.
_EPS = 1e-9


def _parse_box(box):
    """Returns ``(x, y, w, h)`` as floats, or None if malformed or non-finite."""
    if box is None:
        return None

    try:
        if len(box) != 4:
            return None
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None

    return x, y, w, h


def find_degenerate(boxes):
    """Flags boxes with ``w <= 0`` or ``h <= 0``, or any NaN/inf value.

    Missing or malformed boxes (not four numbers) are flagged too.
    """
    flagged = []
    for label_id, box in boxes:
        parsed = _parse_box(box)
        if parsed is None:
            flagged.append(label_id)
            continue

        _, _, w, h = parsed
        if w <= 0 or h <= 0:
            flagged.append(label_id)

    return flagged


def find_out_of_bounds(boxes, tol=0.001):
    """Flags boxes with any edge outside ``[0, 1]`` by more than ``tol``.

    Malformed or non-finite boxes are skipped here; ``find_degenerate`` owns
    them, so each problem gets exactly one tag.
    """
    _check_threshold("tol", tol, 0)
    lo = -tol - _EPS
    hi = 1.0 + tol + _EPS

    flagged = []
    for label_id, box in boxes:
        parsed = _parse_box(box)
        if parsed is None:
            continue

        x, y, w, h = parsed
        edges = (x, y, x + w, y + h)
        if any(e < lo or e > hi for e in edges):
            flagged.append(label_id)

    return flagged


def _parse_size(size):
    """Returns ``(width, height)`` as positive finite floats, or None."""
    try:
        width, height = (float(v) for v in size)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        return None

    return width, height


def _iter_pixel_boxes(boxes):
    """Yields ``(label_id, w_px, h_px)`` for boxes the pixel checks can judge.

    Skips boxes with no usable image size, and boxes that are malformed or have
    ``w <= 0`` or ``h <= 0``: those are owned by ``find_degenerate``.
    """
    for label_id, box, size in boxes:
        parsed = _parse_box(box)
        img = _parse_size(size)
        if parsed is None or img is None:
            continue

        _, _, w, h = parsed
        if w <= 0 or h <= 0:
            continue

        yield label_id, w * img[0], h * img[1]


def _check_threshold(name, value, minimum):
    if not (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum
    ):
        raise ValueError(
            "%s must be a finite number >= %s, got %r" % (name, minimum, value)
        )


def find_tiny(boxes, min_area_px=16):
    """Flags boxes whose pixel area is strictly below ``min_area_px``.

    Area uses the raw box, not clipped to the image.
    """
    _check_threshold("min_area_px", min_area_px, 0)
    limit = min_area_px - _EPS

    return [
        label_id
        for label_id, w_px, h_px in _iter_pixel_boxes(boxes)
        if w_px * h_px < limit
    ]


def find_extreme_aspect_ratio(boxes, max_ratio=20):
    """Flags boxes whose ``w_px / h_px`` is outside ``[1/max_ratio, max_ratio]``.

    The ratio is computed in pixels, so a square box on a wide image is not
    flagged. Ratios exactly at either bound are not flagged.
    """
    _check_threshold("max_ratio", max_ratio, 1)
    hi = max_ratio * (1 + _EPS)
    lo = (1.0 / max_ratio) * (1 - _EPS)

    flagged = []
    for label_id, w_px, h_px in _iter_pixel_boxes(boxes):
        ratio = w_px / h_px
        if ratio > hi or ratio < lo:
            flagged.append(label_id)

    return flagged


def box_iou(a, b):
    """IoU of two ``[x, y, w, h]`` boxes, matching FiftyOne's
    ``fiftyone.utils.iou.compute_bbox_iou`` (non-crowd).

    IoU is scale-invariant, so relative coordinates give the same result as
    pixels. Returns 0 for disjoint or touching boxes.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    iw = min(ax + aw, bx + bw) - max(ax, bx)
    if iw <= 0:
        return 0.0

    ih = min(ay + ah, by + bh) - max(ay, by)
    if ih <= 0:
        return 0.0

    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0

    return min(inter / union, 1.0)


def find_duplicates(labels, iou_thresh=0.95):
    """Flags same-class boxes in ONE sample that overlap another by IoU strictly
    above ``iou_thresh``.

    ``labels`` is an iterable of ``(label_id, class_label, bounding_box,
    confidence)``, with ``confidence`` possibly None.

    Exactly one label of each duplicate group stays unflagged. Labels are ranked
    by confidence (highest first; None ranks last), ties broken by input order.
    Walking that ranking, a label is flagged if it duplicates an already-kept
    label of the same class, otherwise it is kept. So for a pair, the
    lower-confidence label is flagged, or the later one if tied, never both.

    Degenerate boxes are skipped; ``find_degenerate`` owns them.
    """
    _check_threshold("iou_thresh", iou_thresh, 0)
    if iou_thresh > 1:
        raise ValueError("iou_thresh must be <= 1, got %r" % (iou_thresh,))

    limit = iou_thresh + _EPS

    ranked = []
    for index, (label_id, label, box, confidence) in enumerate(labels):
        parsed = _parse_box(box)
        if parsed is None or parsed[2] <= 0 or parsed[3] <= 0:
            continue

        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = None
        if conf is None or not math.isfinite(conf):
            conf = -math.inf

        ranked.append((-conf, index, label_id, label, parsed))

    ranked.sort(key=lambda r: (r[0], r[1]))

    kept = {}  # class_label -> [box, ...]
    flagged = []
    for _, index, label_id, label, box in ranked:
        same_class = kept.setdefault(label, [])
        if any(box_iou(box, other) > limit for other in same_class):
            flagged.append((index, label_id))
        else:
            same_class.append(box)

    return [label_id for _, label_id in sorted(flagged)]


# Relative-coordinate area below which a polygon's convex hull counts as zero.
# 1e-12 is far below one pixel even on a 1M x 1M image.
_ZERO_AREA = 1e-12


def polygon_problem(shape):
    """Returns why one polygon shape is bad, or None if it is fine.

    ``shape`` is a list of ``(x, y)`` points. Reasons:

    -   ``"too_few_points"``: fewer than 3 distinct points
    -   ``"non_finite"``: a coordinate is NaN/inf or not a number
    -   ``"zero_area"``: all points collinear (convex hull has no area)
    -   ``"self_intersecting"``: edges cross or touch (shapely ``is_valid``
        is False)

    A symmetric bow-tie has zero *signed* area, so zero area is judged by the
    convex hull, not the polygon, so that it is reported as self-intersecting.
    """
    from shapely.geometry import MultiPoint, Polygon

    points = []
    try:
        for pt in shape or []:
            x, y = (float(v) for v in pt)
            points.append((x, y))
    except (TypeError, ValueError):
        return "non_finite"

    if not all(math.isfinite(v) for pt in points for v in pt):
        return "non_finite"

    if len(set(points)) < 3:
        return "too_few_points"

    if MultiPoint(points).convex_hull.area < _ZERO_AREA:
        return "zero_area"

    if not Polygon(points).is_valid:
        return "self_intersecting"

    return None


def find_bad_polygons(polylines):
    """Flags polygon labels with any bad shape (see ``polygon_problem``).

    ``polylines`` is an iterable of ``(label_id, points, closed, filled)``,
    where ``points`` is FiftyOne's list of shapes. Only polygons are judged:
    a polyline with ``closed`` and ``filled`` both False is an open line (for
    example a lane marking) and is skipped. A polygon label with no shapes is
    flagged.
    """
    flagged = []
    for label_id, points, closed, filled in polylines:
        if not (closed or filled):
            continue

        shapes = points or []
        if not shapes or any(polygon_problem(s) for s in shapes):
            flagged.append(label_id)

    return flagged


def parse_class_list(text):
    """Splits a comma-separated class list, stripping whitespace around each
    name and dropping empty entries. Class names cannot contain commas.
    """
    if not text:
        return []

    return [name.strip() for name in str(text).split(",") if name.strip()]


def find_unknown_class(labels, allowed):
    """Flags labels whose class is not in ``allowed``.

    ``labels`` is an iterable of ``(label_id, class_label)``. Matching is exact
    and case-sensitive (``"Cat"`` is not ``"cat"``). A missing (None) class is
    flagged. An empty ``allowed`` means no vocabulary was declared, so nothing
    is flagged.
    """
    allowed = set(allowed or [])
    if not allowed:
        return []

    return [label_id for label_id, label in labels if label not in allowed]

