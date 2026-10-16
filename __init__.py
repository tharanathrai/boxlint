"""boxlint: a FiftyOne plugin that lints detection and polygon labels.

Thin glue only: build forms, translate form params into an
``engine.run_checks`` call, and render the report. All check logic lives in
``engine.py`` and ``checks.py``.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import engine

ALL_FIELDS = "__all__"

# param name -> (check_id, min, max, label, description); defaults in engine
THRESHOLDS = {
    "oob_tol": (
        "out_of_bounds",
        0,
        None,
        "Out-of-bounds tolerance",
        "Relative distance an edge may extend past [0, 1]",
    ),
    "min_area_px": (
        "tiny",
        0,
        None,
        "Minimum box area (pixels)",
        "Boxes with a smaller pixel area are flagged",
    ),
    "max_ratio": (
        "aspect_ratio",
        1,
        None,
        "Maximum aspect ratio",
        "Flag boxes wider or taller than this ratio, in pixels",
    ),
    "iou_thresh": (
        "duplicate",
        0,
        1,
        "Duplicate IoU threshold",
        "Same-class boxes with IoU strictly above this are duplicates",
    ),
}

FILTER_HINT = (
    "Open the sidebar, expand label tags, and select an issue:* tag to view "
    "flagged labels"
)


class CheckLabels(foo.Operator):
    # Python API: foo.get_operator(uri).run_checks(view, field, ...)
    run_checks = staticmethod(engine.run_checks)

    @property
    def config(self):
        return foo.OperatorConfig(
            name="check_labels",
            label="Check labels",
            description=(
                "Scan detection and polygon labels for structural problems "
                "and tag them with issue:* tags"
            ),
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Resolving the target here opts the form into FiftyOne's built-in
        # dataset / current view picker
        view = ctx.target_view()

        field_kinds = engine.get_label_fields(ctx.dataset)
        if not field_kinds:
            _no_fields_warning(inputs)
            return types.Property(inputs)

        field = _field_input(inputs, ctx, field_kinds)
        kind = field_kinds[field]

        for check_id in engine.applicable_checks(kind):
            if check_id in engine.VOCAB_CHECKS:
                continue

            _, severity, label, description = engine.CHECKS[check_id]
            inputs.bool(
                "check_" + check_id,
                default=True,
                label="%s (%s)" % (label, severity),
                description=description,
                view=types.CheckboxView(),
            )

            # Each threshold sits directly under its own checkbox
            if not _is_enabled(ctx.params, check_id):
                continue

            for param, (t_check, lo, hi, t_label, t_desc) in THRESHOLDS.items():
                if t_check == check_id:
                    inputs.float(
                        param,
                        default=engine.DEFAULTS[param],
                        min=lo,
                        max=hi,
                        label=t_label,
                        description=t_desc,
                    )

        if kind == engine.DETECTIONS:
            inputs.str(
                "allowed_classes",
                default=", ".join(engine.declared_classes(ctx.dataset, field)),
                label="Allowed classes (error)",
                description=(
                    "Optional comma-separated class list; labels with any "
                    "other class are tagged issue:unknown_class. Leave empty "
                    "to skip. Pre-filled from the dataset's declared classes, "
                    "if any"
                ),
            )

            pixel_on = any(
                _is_enabled(ctx.params, c) for c in engine.PIXEL_CHECKS
            )
            num_missing = engine.count_missing_metadata(view) if pixel_on else 0
            if num_missing:
                inputs.view(
                    "metadata_warning",
                    types.Warning(
                        label="Image size missing",
                        description=(
                            "%d of %d samples have no image width/height. "
                            "Tiny and aspect ratio checks need it."
                            % (num_missing, view.count())
                        ),
                    ),
                )
                inputs.bool(
                    "compute_metadata",
                    default=True,
                    label="Compute missing image metadata",
                    description=(
                        "Reads each image header to get width/height. If off, "
                        "pixel checks skip these samples and the output says so"
                    ),
                    view=types.CheckboxView(),
                )

        inputs.bool(
            "clear_previous",
            default=True,
            label="Clear previous issue:* tags first",
            description=(
                "Recommended. If off, tags from earlier runs stay, so results "
                "may mix old and new findings"
            ),
            view=types.CheckboxView(),
        )

        return types.Property(inputs)

    def execute(self, ctx):
        params = ctx.params
        view = ctx.target_view()
        field = params["label_field"]
        kind = engine.get_label_fields(view)[field]

        allowed = params.get("allowed_classes") or ""
        enabled = [
            c
            for c in engine.applicable_checks(kind)
            if c not in engine.VOCAB_CHECKS and _is_enabled(params, c)
        ]
        if kind == engine.DETECTIONS and engine.parse_allowed_classes(allowed):
            enabled.append("unknown_class")

        report = engine.run_checks(
            view,
            field,
            checks=enabled,
            allowed_classes=allowed,
            compute_metadata=params.get("compute_metadata", True),
            clear_previous=params.get("clear_previous", True),
            tag=True,
            **{p: _threshold(params, p) for p in THRESHOLDS},
        )

        ctx.ops.reload_dataset()

        return _report_to_output(report)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("label_field", label="Label field")

        table = types.TableView()
        table.add_column("check", label="Check")
        table.add_column("severity", label="Severity")
        table.add_column("labels", label="Flagged labels")
        table.add_column("samples", label="Flagged samples")
        table.add_column("top_classes", label="Top classes")
        table.add_column("details", label="Details")
        outputs.list("summary", types.Object(), label="Summary", view=table)

        outputs.int("num_labels_checked", label="Labels checked")
        outputs.int("num_error_labels", label="Labels with errors")
        outputs.int("num_warning_labels", label="Labels with warnings")
        outputs.int("num_flagged_samples", label="Samples with any flag")
        outputs.int("num_tags_cleared", label="Previous issue:* tags cleared")
        outputs.str("skipped", label="Skipped")
        outputs.str("message", label="Next step")
        return types.Property(outputs)


class ClearIssueTags(foo.Operator):
    # Python API: foo.get_operator(uri).clear_issue_tags(view, fields)
    clear_issue_tags = staticmethod(engine.clear_issue_tags)

    @property
    def config(self):
        return foo.OperatorConfig(
            name="clear_issue_tags",
            label="Clear issue tags",
            description="Remove issue:* tags written by Check labels",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        ctx.target_view()

        field_kinds = engine.get_label_fields(ctx.dataset)
        if not field_kinds:
            _no_fields_warning(inputs)
            return types.Property(inputs)

        _field_input(inputs, ctx, field_kinds, allow_all=True)
        return types.Property(inputs)

    def execute(self, ctx):
        view = ctx.target_view()
        field = ctx.params.get("label_field") or ALL_FIELDS
        if field == ALL_FIELDS:
            fields = sorted(engine.get_label_fields(view))
        else:
            fields = [field]

        removed = engine.clear_issue_tags(view, fields)
        ctx.ops.reload_dataset()

        return {
            "fields": ", ".join(fields),
            "removed": [
                {"tag": t, "count": n} for t, n in sorted(removed.items())
            ],
            "total": sum(removed.values()),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("fields", label="Fields")
        table = types.TableView()
        table.add_column("tag", label="Tag")
        table.add_column("count", label="Labels untagged")
        outputs.list("removed", types.Object(), label="Removed", view=table)
        outputs.int("total", label="Total tags removed")
        return types.Property(outputs)


def _no_fields_warning(inputs):
    inputs.view(
        "no_fields",
        types.Warning(
            label="No Detections or Polylines fields",
            description="This dataset has no label fields to check",
        ),
    )


def _field_input(inputs, ctx, field_kinds, allow_all=False):
    """Adds the label field dropdown; returns the currently selected field."""
    fields = sorted(field_kinds)
    choices = types.Dropdown()
    if allow_all:
        choices.add_choice(ALL_FIELDS, label="All Detections/Polylines fields")
    for field in fields:
        choices.add_choice(field, label=field)

    default = ALL_FIELDS if allow_all else fields[0]
    inputs.enum(
        "label_field",
        choices.values(),
        default=default,
        required=True,
        label="Label field",
        view=choices,
    )
    return ctx.params.get("label_field") or default


def _is_enabled(params, check_id):
    return bool(params.get("check_" + check_id, True))


def _threshold(params, param):
    value = params.get(param)
    return engine.DEFAULTS[param] if value is None else value


def _report_to_output(report):
    rows = []
    for check_id, result in report["checks"].items():
        counts = result["class_counts"]
        top = ", ".join(
            "%s (%d)" % (c, counts[c]) for c in result["top_classes"]
        )
        details = ", ".join(
            "%s (%d)" % kv for kv in result.get("reason_counts", {}).items()
        )
        rows.append(
            {
                "check": engine.CHECKS[check_id][2],
                "severity": result["severity"],
                "labels": result["num_labels"],
                "samples": result["num_samples"],
                "top_classes": top or "-",
                "details": details or "-",
            }
        )

    return {
        "label_field": report["label_field"],
        "summary": rows,
        "num_labels_checked": report["num_labels_checked"],
        "num_error_labels": report["num_error_labels"],
        "num_warning_labels": report["num_warning_labels"],
        "num_flagged_samples": report["num_flagged_samples"],
        "num_tags_cleared": report["num_tags_cleared"],
        "skipped": "; ".join(report["skipped"]) or "None",
        "message": FILTER_HINT,
    }


def register(p):
    p.register(CheckLabels)
    p.register(ClearIssueTags)
