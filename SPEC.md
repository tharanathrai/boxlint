# SPEC: boxlint

## Problem

Detection and polygon datasets pass through annotation tools, vendors, converters, resizers, and
merges. Each step can corrupt geometry. Bad labels fail far from where they were made: depending
on the framework they crash training, turn the loss into NaN, or are silently dropped, and
evaluation gets distorted. This plugin provides a fast, deterministic audit that
runs inside the FiftyOne App and from Python.

Positioning: a model-free **label linter**. It answers "does this label break a structural or
declared rule?", not "does a model think this label is wrong?" (that is the Brain's mistakenness).

## Goals

1. Detect structural label problems with one operator run.
2. Make results actionable inside the App (filterable tags, summary, jump to flagged view).
3. Be safe: never modify or delete labels.
4. Be small and easy to extend: one function per check.
5. Usable outside the App: the same engine runs from Python and returns a machine-readable report,
   so it can gate dataset imports and releases.

## Non-goals

- Semantic correctness (wrong class). The Brain's mistakenness covers that.
- Automatic fixing.
- Classification-only datasets.

## Checks

| ID | Tag | Severity | Applies to | Rule | Default |
|---|---|---|---|---|---|
| degenerate | `issue:degenerate` | error | Detection | `w <= 0` or `h <= 0` or any value NaN/inf | on |
| out_of_bounds | `issue:out_of_bounds` | error | Detection | any edge outside [0, 1] beyond tolerance | on, tol 0.001 |
| tiny | `issue:tiny` | warning | Detection | pixel area below `min_area_px` (needs metadata) | on, 16 px area |
| aspect_ratio | `issue:aspect_ratio` | warning | Detection | `w_px / h_px` outside `[1/max_ratio, max_ratio]` | on, max 20 |
| duplicate | `issue:duplicate` | warning | Detection | same-sample, same-class IoU above threshold with another label; tag the lower-confidence or later one, not both | on, IoU 0.95 |
| unknown_class | `issue:unknown_class` | error | Detection | class label not in the user-supplied allowed list (exact, case-sensitive) | off until a list is given (milestone 4b) |
| bad_polygon | `issue:bad_polygon` | error | Polyline | no shapes, or a shape with fewer than 3 points, zero area, NaN/inf, or crossing/overlapping edges (shapely). A ring touching itself at one point (pinch) is fine; degenerate slivers are ignored when the label has another valid shape | on (milestone 4) |

Notes:
- Pixel-based checks (tiny, aspect_ratio) need image dimensions. If metadata is missing, offer to
  compute it, or skip those checks and say so in the output. Never silently skip.
- Duplicate IoU: verify whether `fiftyone.utils.iou` provides what is needed before writing a custom
  implementation. A simple O(n^2) per sample is acceptable since n is small per image.
- Multiple issues on one label produce multiple tags, except that malformed/NaN/non-positive boxes
  are owned by `degenerate` only; other checks skip them so one problem gets one tag.
- Severity: `error` means the label violates a hard or declared rule; `warning` means unusual and
  worth review, not necessarily wrong. Severity appears in the summary and the report; tags do
  not encode it.
- Duplicate rule (implemented): rank labels by confidence (missing last), ties by order; walk the
  ranking and flag a label if it overlaps an already-kept same-class label by IoU > threshold.
  One label per duplicate group always stays unflagged. FiftyOne's own recipe
  (https://docs.voxel51.com/recipes/remove_duplicate_annos.html) flags the later label of each
  pair instead; the README should link it and note the difference.

## Operator: `check_labels`

Inputs (form):
- Label field (dropdown of Detections / Polylines fields on the dataset)
- Target: entire dataset or current view
- Checkboxes per check, with threshold inputs shown only when the check is enabled
- Option: clear previous `issue:*` tags first (default true)
- Execution: allow delegated execution for large datasets

Execution:
1. Resolve target view and label field.
2. If clearing, untag all existing `issue:*` tags on that field.
3. Run each enabled check, collecting `{check_id: [label_id, ...]}` and per-class counts.
4. Tag labels per check via `select_labels` + `tag_labels`.
5. Return a results dict.

Output:
- Table: check, severity, flagged labels, flagged samples, top 3 classes affected, details
  (polygon failure reasons).
- Totals: labels checked, labels with errors, labels with warnings, samples with any flag,
  previous tags cleared.
- Message with how to filter: "Open the sidebar, expand label tags, select an `issue:*` tag."
- A line listing any checks skipped and why.

## Python API

```python
import fiftyone.operators as foo

op = foo.get_operator("@tharanathrai/boxlint/check_labels")
report = op.run_checks(
    view, "ground_truth",
    checks=None,               # default: all checks for the field type
    min_area_px=16, max_ratio=20, oob_tol=0.001, iou_thresh=0.95,
    allowed_classes=None,      # list or comma-separated string; enables unknown_class
    compute_metadata=True, clear_previous=True,
    tag=True,                  # False = dry run, writes no tags
)
```

- Implemented in `engine.py` (`run_checks`); the operator exposes it as a method, so callers
  need only the operator URI, not the plugin's internal module path (FiftyOne imports plugins
  as `fiftyone.plugins.orgs.<org>.<name>`, which depends on the plugin name).
- The operator's `execute` calls this same function, so App and Python results match.
- All inputs (field, check IDs, thresholds) are validated before anything is written.
- Report keys: `label_field`, `field_type`, `num_samples`, `num_labels_checked`,
  `num_flagged_labels`, `num_flagged_samples`, `num_error_labels`, `num_warning_labels`,
  `num_tags_cleared`, `skipped`, `config`, and `checks.<id>` with `severity`, `tag`,
  `num_labels`, `num_samples`, `class_counts`, `top_classes`, `label_ids` (plus
  `reason_counts` for `bad_polygon`). JSON-serializable.
- `clear_issue_tags` operator exposes `engine.clear_issue_tags(view, fields)` the same way.

## Second operator (stretch): `clear_issue_tags`

Removes all `issue:*` tags from a chosen field.

## Acceptance criteria

- On `quickstart`, with deliberately injected bad labels (a zero-width box, a box with x+w > 1, a
  duplicated box), each is flagged by the correct check and nothing else is flagged.
- Re-running produces identical tag counts (idempotent).
- A clean dataset yields zero flags with default settings (no false positives on `quickstart`
  beyond a small, explainable number; document any).
- Unit tests cover every check, including boundary values.
- `run_checks` from Python gives the same counts as the operator on the same view.
- README has a GIF or screenshots and a quickstart.

## Out of scope for v1 (v2 candidates)

- Versioned ruleset files (allowed classes, required attributes, geometry policies).
- Required/enum attribute checks and cross-field consistency checks.
- Review states (confirmed issue / acceptable exception / resolved). Conflicts with clearing all
  `issue:*` tags on every run; needs a separate tag namespace and a "skip excepted" option.
- A custom panel; operator output covers v1.
- Model-assisted signals. If ever added, reuse Brain mistakenness and report it separately from
  rule violations.

## Open questions (resolve by reading docs or source, then note the answer here)

- Best way to tag a specific list of label IDs efficiently on large datasets.
  Answer: `view.select_labels(ids=..., fields=field).tag_labels(tag, label_fields=field)`, one
  call per check. Pulling geometry with one `values()` call per field keeps it fast: 3,000
  samples / 18,585 boxes check and tag in 0.4 s; 18,540 polygons in 1.0 s.
- Whether the output can render a table natively via the operator output types.
  Answer: yes, `types.TableView` with `add_column`, on an `outputs.list(..., types.Object())`.
- Community channel for the launch post. Answer: Discord `#plugins`, per the fiftyone-plugins
  README (no Slack mentioned).
- Whether the output can render a table natively via the operator output types.
- Behavior of `compute_metadata` on remote or very large media.
