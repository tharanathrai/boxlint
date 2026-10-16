import math

import pytest

from checks import (
    box_iou,
    find_degenerate,
    find_duplicates,
    find_bad_polygons,
    find_extreme_aspect_ratio,
    find_out_of_bounds,
    find_tiny,
    find_unknown_class,
    parse_class_list,
    polygon_label_problem,
    polygon_problem,
)

NAN = float("nan")
INF = float("inf")


def _ids(*boxes):
    return [(str(i), b) for i, b in enumerate(boxes)]


# --- find_degenerate ---------------------------------------------------------


def test_degenerate_empty():
    assert find_degenerate([]) == []


def test_degenerate_valid_box_not_flagged():
    assert find_degenerate(_ids([0.1, 0.1, 0.2, 0.3])) == []


@pytest.mark.parametrize(
    "box",
    [
        [0.1, 0.1, 0.0, 0.3],  # w == 0
        [0.1, 0.1, 0.2, 0.0],  # h == 0
        [0.1, 0.1, -0.1, 0.3],  # w < 0
        [0.1, 0.1, 0.2, -1e-6],  # h < 0
        [NAN, 0.1, 0.2, 0.3],
        [0.1, 0.1, 0.2, NAN],
        [0.1, INF, 0.2, 0.3],
        [0.1, 0.1, -INF, 0.3],
    ],
)
def test_degenerate_flags(box):
    assert find_degenerate(_ids(box)) == ["0"]


@pytest.mark.parametrize("box", [None, [], [0.1, 0.1, 0.2], [0.1] * 5, ["a", 0, 1, 1]])
def test_degenerate_flags_malformed(box):
    assert find_degenerate(_ids(box)) == ["0"]


def test_degenerate_tiny_positive_not_flagged():
    # Tiny is a separate check; degenerate only means non-positive
    assert find_degenerate(_ids([0.5, 0.5, 1e-12, 1e-12])) == []


def test_degenerate_preserves_order_and_ids():
    boxes = [("a", [0, 0, 0, 1]), ("b", [0, 0, 1, 1]), ("c", [0, 0, 1, NAN])]
    assert find_degenerate(boxes) == ["a", "c"]


# --- find_out_of_bounds ------------------------------------------------------


def test_oob_empty():
    assert find_out_of_bounds([]) == []


@pytest.mark.parametrize(
    "box",
    [
        [0.0, 0.0, 1.0, 1.0],  # full image, exactly on all edges
        [0.5, 0.5, 0.5, 0.5],  # x + w == 1.0 exactly
        [0.3, 0.3, 0.7, 0.7],  # x + w == 1.0 up to float rounding
        [0.0, 0.0, 1.0005, 0.5],  # 0.0005 over, within default tol 0.001
        [-0.0005, 0.0, 0.5, 0.5],  # 0.0005 under, within tol
        [0.0, 0.0, 1.001, 1.001],  # exactly at tol
        [-0.001, -0.001, 0.5, 0.5],  # exactly at -tol
    ],
)
def test_oob_not_flagged(box):
    assert find_out_of_bounds(_ids(box)) == []


@pytest.mark.parametrize(
    "box",
    [
        [0.5, 0.5, 0.6, 0.2],  # x + w == 1.1
        [0.0, 0.5, 0.2, 0.6],  # y + h == 1.1
        [-0.01, 0.0, 0.5, 0.5],  # x < 0
        [0.0, -0.01, 0.5, 0.5],  # y < 0
        [0.0, 0.0, 1.0015, 0.5],  # just past tol
        [1.5, 1.5, 0.1, 0.1],  # entirely outside
    ],
)
def test_oob_flags(box):
    assert find_out_of_bounds(_ids(box)) == ["0"]


def test_oob_zero_tol():
    assert find_out_of_bounds(_ids([0.0, 0.0, 1.0005, 0.5]), tol=0) == ["0"]
    assert find_out_of_bounds(_ids([0.0, 0.0, 1.0, 1.0]), tol=0) == []


def test_oob_larger_tol():
    assert find_out_of_bounds(_ids([0.5, 0.5, 0.6, 0.2]), tol=0.2) == []


@pytest.mark.parametrize("box", [None, [0.5, 0.5, NAN, 0.2], [INF, 0, 1, 1], [1, 2]])
def test_oob_skips_malformed(box):
    # Owned by find_degenerate, so no double-tagging
    assert find_out_of_bounds(_ids(box)) == []


def test_oob_zero_width_inside_not_flagged():
    # A zero-width box inside the image is degenerate, not out of bounds
    assert find_out_of_bounds(_ids([0.5, 0.5, 0.0, 0.2])) == []


@pytest.mark.parametrize("tol", [-0.001, NAN, INF, None, "0.1"])
def test_oob_rejects_bad_tol(tol):
    with pytest.raises(ValueError):
        find_out_of_bounds([], tol=tol)


def test_eps_is_tiny():
    from checks import _EPS

    assert 0 < _EPS < 1e-6 and not math.isnan(_EPS)


# --- pixel checks: shared helpers ------------------------------------------

SIZE = (100, 100)


def _px(*boxes, size=SIZE):
    return [(str(i), b, size) for i, b in enumerate(boxes)]


# --- find_tiny ---------------------------------------------------------------


def test_tiny_empty():
    assert find_tiny([]) == []


def test_tiny_exactly_at_threshold_not_flagged():
    # 4 px x 4 px = 16 px area on a 100 x 100 image
    assert find_tiny(_px([0.1, 0.1, 0.04, 0.04])) == []


def test_tiny_just_below_threshold_flagged():
    assert find_tiny(_px([0.1, 0.1, 0.04, 0.0399])) == ["0"]


def test_tiny_uses_pixels_not_relative_area():
    box = [0.1, 0.1, 0.01, 0.01]  # 1e-4 relative area
    assert find_tiny(_px(box, size=(100, 100))) == ["0"]  # 1 px area
    assert find_tiny(_px(box, size=(4000, 4000))) == []  # 1600 px area


def test_tiny_threshold_changes_result():
    box = [0.1, 0.1, 0.1, 0.1]  # 100 px area
    assert find_tiny(_px(box), min_area_px=100) == []
    assert find_tiny(_px(box), min_area_px=101) == ["0"]
    assert find_tiny(_px(box), min_area_px=0) == []


@pytest.mark.parametrize("box", [[0.1, 0.1, 0.0, 0.1], [0.1, 0.1, 0.1, -0.1], [NAN, 0, 0.1, 0.1], None])
def test_tiny_skips_degenerate(box):
    # Owned by find_degenerate, so a zero-width box gets one tag, not two
    assert find_tiny(_px(box)) == []


@pytest.mark.parametrize("size", [None, (None, None), (0, 100), (100, -1), (NAN, 100), (100,), "ab"])
def test_tiny_skips_unknown_image_size(size):
    assert find_tiny(_px([0.1, 0.1, 0.001, 0.001], size=size)) == []


@pytest.mark.parametrize("value", [-1, NAN, INF, None, True, "16"])
def test_tiny_rejects_bad_threshold(value):
    with pytest.raises(ValueError):
        find_tiny([], min_area_px=value)


# --- find_extreme_aspect_ratio -----------------------------------------------


def test_aspect_empty():
    assert find_extreme_aspect_ratio([]) == []


@pytest.mark.parametrize(
    "box",
    [
        [0.0, 0.0, 0.4, 0.4],  # square
        [0.0, 0.0, 0.8, 0.04],  # exactly 20:1
        [0.0, 0.0, 0.04, 0.8],  # exactly 1:20
    ],
)
def test_aspect_not_flagged(box):
    assert find_extreme_aspect_ratio(_px(box)) == []


@pytest.mark.parametrize(
    "box",
    [
        [0.0, 0.0, 0.81, 0.04],  # 20.25:1
        [0.0, 0.0, 0.04, 0.81],  # 1:20.25
        [0.0, 0.0, 1.0, 0.001],  # 1000:1
    ],
)
def test_aspect_flagged(box):
    assert find_extreme_aspect_ratio(_px(box)) == ["0"]


def test_aspect_uses_pixel_ratio():
    # Relative 1:1 box on a 2000 x 100 image is 20:1 in pixels (on the bound)
    box = [0.0, 0.0, 0.5, 0.5]
    assert find_extreme_aspect_ratio(_px(box, size=(2000, 100))) == []
    assert find_extreme_aspect_ratio(_px(box, size=(2100, 100))) == ["0"]
    # Relative 20:1 box on a 1 x 20 image is 1:1 in pixels
    assert find_extreme_aspect_ratio(_px([0, 0, 1.0, 0.05], size=(1, 20))) == []


def test_aspect_threshold_changes_result():
    box = [0.0, 0.0, 0.5, 0.1]  # 5:1
    assert find_extreme_aspect_ratio(_px(box), max_ratio=5) == []
    assert find_extreme_aspect_ratio(_px(box), max_ratio=4.9) == ["0"]
    # max_ratio == 1 flags everything that is not square
    assert find_extreme_aspect_ratio(_px(box, [0, 0, 0.2, 0.2]), max_ratio=1) == ["0"]


@pytest.mark.parametrize("box", [[0.1, 0.1, 0.5, 0.0], [0.1, 0.1, 0.0, 0.5], [0.1, 0.1, INF, 0.1]])
def test_aspect_skips_degenerate(box):
    # No division by zero, and no double tag for degenerate boxes
    assert find_extreme_aspect_ratio(_px(box)) == []


def test_aspect_skips_unknown_image_size():
    assert find_extreme_aspect_ratio(_px([0, 0, 1.0, 0.001], size=None)) == []


@pytest.mark.parametrize("value", [0.5, 0, -20, NAN, INF, None, "20"])
def test_aspect_rejects_bad_threshold(value):
    with pytest.raises(ValueError):
        find_extreme_aspect_ratio([], max_ratio=value)


# --- box_iou / find_duplicates -----------------------------------------------


def _dets(*items):
    # items: (class_label, box) or (class_label, box, confidence)
    out = []
    for i, item in enumerate(items):
        label, box = item[0], item[1]
        conf = item[2] if len(item) > 2 else None
        out.append((str(i), label, box, conf))
    return out


A = [0.1, 0.1, 0.4, 0.4]


def test_iou_basic_values():
    assert box_iou(A, A) == 1.0
    assert box_iou([0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]) == 0.0  # touching corner
    assert box_iou([0, 0, 0.5, 1], [0.5, 0, 0.5, 1]) == 0.0  # touching edge
    assert box_iou([0, 0, 1, 1], [0, 0, 0.5, 1]) == pytest.approx(0.5)


def test_iou_matches_fiftyone():
    fo = pytest.importorskip("fiftyone")
    foui = pytest.importorskip("fiftyone.utils.iou")
    import random

    rng = random.Random(0)
    for _ in range(500):
        a = [rng.random() * 0.8, rng.random() * 0.8, rng.random() * 0.5 + 0.01, rng.random() * 0.5 + 0.01]
        b = [rng.random() * 0.8, rng.random() * 0.8, rng.random() * 0.5 + 0.01, rng.random() * 0.5 + 0.01]
        expected = foui.compute_bbox_iou(fo.Detection(bounding_box=a), fo.Detection(bounding_box=b))
        assert box_iou(a, b) == pytest.approx(expected, abs=1e-12)


def test_dup_empty_and_single():
    assert find_duplicates([]) == []
    assert find_duplicates(_dets(("cat", A))) == []


def test_dup_identical_tags_later_only():
    assert find_duplicates(_dets(("cat", A), ("cat", A))) == ["1"]


def test_dup_near_identical_flagged():
    near = [0.1, 0.1, 0.4, 0.405]  # IoU ~0.988
    assert find_duplicates(_dets(("cat", A), ("cat", near))) == ["1"]


def test_dup_different_classes_not_flagged():
    assert find_duplicates(_dets(("cat", A), ("dog", A))) == []


def test_dup_below_threshold_not_flagged():
    shifted = [0.1, 0.1, 0.4, 0.5]  # IoU 0.8
    assert find_duplicates(_dets(("cat", A), ("cat", shifted))) == []


def test_dup_exactly_at_threshold_not_flagged():
    # Nested box covering 95% of the outer one: IoU == 0.95 exactly
    items = _dets(("cat", [0, 0, 1, 1]), ("cat", [0, 0, 0.95, 1]))
    assert find_duplicates(items, iou_thresh=0.95) == []
    assert find_duplicates(items, iou_thresh=0.949) == ["1"]


def test_dup_tags_lower_confidence_not_later():
    items = _dets(("cat", A, 0.3), ("cat", A, 0.9))
    assert find_duplicates(items) == ["0"]


def test_dup_equal_confidence_tags_later():
    assert find_duplicates(_dets(("cat", A, 0.5), ("cat", A, 0.5))) == ["1"]


def test_dup_missing_confidence_ranks_last():
    assert find_duplicates(_dets(("cat", A, None), ("cat", A, 0.1))) == ["0"]
    assert find_duplicates(_dets(("cat", A, NAN), ("cat", A, 0.1))) == ["0"]


def test_dup_cluster_of_three_keeps_one():
    assert find_duplicates(_dets(("cat", A), ("cat", A), ("cat", A))) == ["1", "2"]


def test_dup_chain_keeps_unrelated_end():
    # 0~1 and 1~2 are duplicates, 0 and 2 are not: only 1 is flagged
    b0 = [0.0, 0.0, 0.5, 0.5]
    b1 = [0.0, 0.0, 0.5, 0.52]  # IoU(b0, b1) ~0.96
    b2 = [0.0, 0.0, 0.5, 0.545]  # IoU(b1, b2) ~0.954, IoU(b0, b2) ~0.917
    assert box_iou(b0, b2) < 0.95 < min(box_iou(b0, b1), box_iou(b1, b2))
    assert find_duplicates(_dets(("cat", b0), ("cat", b1), ("cat", b2))) == ["1"]


def test_dup_skips_degenerate():
    zero = [0.1, 0.1, 0.0, 0.4]
    items = _dets(("cat", zero), ("cat", zero), ("cat", [NAN, 0, 1, 1]), ("cat", None))
    assert find_duplicates(items) == []


def test_dup_output_in_input_order():
    items = _dets(("cat", A, 0.1), ("dog", A, 0.9), ("cat", A, 0.9), ("dog", A, 0.1))
    assert find_duplicates(items) == ["0", "3"]


@pytest.mark.parametrize("value", [-0.1, 1.01, NAN, None, "0.9"])
def test_dup_rejects_bad_threshold(value):
    with pytest.raises(ValueError):
        find_duplicates([], iou_thresh=value)


def test_dup_threshold_one_flags_nothing():
    assert find_duplicates(_dets(("cat", A), ("cat", A)), iou_thresh=1) == []


# --- polygons ----------------------------------------------------------------

SQUARE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
BOWTIE = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
COLLINEAR = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
PINCH = [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 1)]
SLIVER = [(0.2, 0.2), (0.3, 0.3), (0.2, 0.2)]  # 3-point out-and-back


@pytest.mark.parametrize(
    "shape, reason",
    [
        (SQUARE, None),
        (SQUARE + [SQUARE[0]], None),  # explicitly closed ring
        ([(0.1, 0.1), (0.2, 0.1), (0.1, 0.2)], None),  # triangle, the minimum
        ([(0, 0), (0.5, 0), (0.5, 0.5), (0.25, 0.1), (0, 0.5)], None),  # concave is fine
        (BOWTIE, "self_intersecting"),
        ([(0, 0), (1, 1), (1, 0.2), (0, 0.6)], "self_intersecting"),  # asymmetric bow-tie
        (PINCH, None),  # ring touches itself at one vertex: shape is fine
        ([(0, 0), (2, 0), (2, 2), (1, 2), (1, 0), (0, 2)], None),  # vertex touches an edge
        ([(0, 0), (2, 0), (2, 2), (1, 2), (1, 3), (1, 2), (0, 2)], "self_intersecting"),  # spike
        (COLLINEAR, "zero_area"),
        ([(0, 0), (0.5, 0), (1, 0), (0.25, 0)], "zero_area"),  # 4 collinear points
        ([], "too_few_points"),
        ([(0.1, 0.1), (0.2, 0.2)], "too_few_points"),
        ([(0.1, 0.1), (0.2, 0.2), (0.1, 0.1)], "too_few_points"),  # 3 points, 2 distinct
        ([(0.1, 0.1), (NAN, 0.2), (0.3, 0.1)], "non_finite"),
        ([(0.1, 0.1), (0.2, INF), (0.3, 0.1)], "non_finite"),
        ([(0.1, 0.1), (0.2,), (0.3, 0.1)], "non_finite"),  # malformed point
        (None, "too_few_points"),
    ],
)
def test_polygon_problem(shape, reason):
    assert polygon_problem(shape) == reason


def test_polygon_tiny_but_real_area_not_zero():
    # ~1 px triangle on a 640 x 480 image: area 1.6e-6, well above the cutoff
    tri = [(0.5, 0.5), (0.5 + 1 / 640, 0.5), (0.5, 0.5 + 2 / 480)]
    assert polygon_problem(tri) is None


def _polys(*items):
    # items: (points, closed, filled)
    return [(str(i), pts, c, f) for i, (pts, c, f) in enumerate(items)]


def test_bad_polygons_empty():
    assert find_bad_polygons([]) == []


def test_bad_polygons_flags_bowtie_and_collinear():
    items = _polys(([SQUARE], True, True), ([BOWTIE], True, True), ([COLLINEAR], True, False))
    assert find_bad_polygons(items) == ["1", "2"]


def test_bad_polygons_closed_or_filled_is_enough():
    assert find_bad_polygons(_polys(([BOWTIE], True, False))) == ["0"]
    assert find_bad_polygons(_polys(([BOWTIE], False, True))) == ["0"]


def test_bad_polygons_skips_open_polylines():
    # A zigzag lane line crosses nothing meaningful as a polygon: not judged
    zigzag = [(0, 0), (1, 1), (1, 0), (0, 1)]
    assert find_bad_polygons(_polys(([zigzag], False, False), ([[(0, 0)]], False, False))) == []


def test_bad_polygons_multi_shape_any_bad_flags():
    assert find_bad_polygons(_polys(([SQUARE, SQUARE], True, True))) == []
    assert find_bad_polygons(_polys(([SQUARE, BOWTIE], True, True))) == ["0"]


@pytest.mark.parametrize("points", [[], None, [[]]])
def test_bad_polygons_no_shapes_flagged(points):
    assert find_bad_polygons(_polys((points, True, True))) == ["0"]


# --- allowed classes ---------------------------------------------------------


def _labels(*names):
    return [(str(i), n) for i, n in enumerate(names)]


def test_unknown_class_empty_inputs():
    assert find_unknown_class([], ["cat"]) == []
    assert find_unknown_class(_labels("cat"), []) == []


@pytest.mark.parametrize("allowed", [[], None, set(), ()])
def test_unknown_class_no_vocabulary_flags_nothing(allowed):
    assert find_unknown_class(_labels("cat", "zebra", None), allowed) == []


def test_unknown_class_exact_match():
    assert find_unknown_class(_labels("cat", "dog", "zebra"), ["cat", "dog"]) == ["2"]


def test_unknown_class_case_sensitive():
    assert find_unknown_class(_labels("cat", "Cat", "CAT"), ["cat"]) == ["1", "2"]


def test_unknown_class_whitespace_is_not_ignored_in_labels():
    assert find_unknown_class(_labels("cat", " cat", "cat "), ["cat"]) == ["1", "2"]


def test_unknown_class_none_label_flagged():
    assert find_unknown_class(_labels("cat", None, ""), ["cat"]) == ["1", "2"]


def test_unknown_class_names_with_spaces():
    assert find_unknown_class(_labels("traffic light", "traffic"), ["traffic light"]) == ["1"]


def test_unknown_class_accepts_any_iterable():
    assert find_unknown_class(_labels("a", "b"), ("a",)) == ["1"]
    assert find_unknown_class(_labels("a", "b"), {"b"}) == ["0"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cat,dog", ["cat", "dog"]),
        (" cat , dog ,", ["cat", "dog"]),
        ("traffic light, stop sign", ["traffic light", "stop sign"]),
        ("", []),
        (None, []),
        (" , ,", []),
        ("Cat", ["Cat"]),  # case kept
    ],
)
def test_parse_class_list(text, expected):
    assert parse_class_list(text) == expected


def test_bad_polygons_pinch_not_flagged():
    assert find_bad_polygons(_polys(([PINCH], True, True))) == []


@pytest.mark.parametrize("sliver", [SLIVER, COLLINEAR, [(0.1, 0.1)]])
def test_bad_polygons_sliver_in_multi_shape_ignored(sliver):
    assert find_bad_polygons(_polys(([SQUARE, sliver], True, True))) == []
    assert find_bad_polygons(_polys(([sliver, SQUARE, sliver], True, True))) == []


def test_bad_polygons_all_slivers_flagged():
    assert find_bad_polygons(_polys(([SLIVER, COLLINEAR], True, True))) == ["0"]
    assert find_bad_polygons(_polys(([SLIVER], True, True))) == ["0"]


def test_bad_polygons_crossing_or_nan_in_multi_shape_still_flagged():
    nan_tri = [(0.1, 0.1), (NAN, 0.2), (0.3, 0.1)]
    assert find_bad_polygons(_polys(([SQUARE, SLIVER, BOWTIE], True, True))) == ["0"]
    assert find_bad_polygons(_polys(([SQUARE, nan_tri], True, True))) == ["0"]


@pytest.mark.parametrize(
    "shapes, reason",
    [
        ([], "no_shapes"),
        (None, "no_shapes"),
        ([SQUARE], None),
        ([SQUARE, SLIVER], None),
        ([SLIVER, SQUARE, BOWTIE], "self_intersecting"),  # sliver skipped, crossing reported
        ([SLIVER, COLLINEAR], "too_few_points"),
        ([COLLINEAR], "zero_area"),
        ([PINCH, SLIVER], None),
    ],
)
def test_polygon_label_problem(shapes, reason):
    assert polygon_label_problem(shapes) == reason
