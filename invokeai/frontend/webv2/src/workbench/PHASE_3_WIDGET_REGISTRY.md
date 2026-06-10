# Phase 3 Widget Registry

Phase 3 establishes the registration and graph-bearing surface contracts.

## Widget Manifest

`WidgetManifest` declares:

- stable widget id and label
- manifest version
- allowed workbench regions
- icon key
- availability policy: enabled, disabled, or hidden
- optional graph-bearing source metadata
- registration/render failure behavior
- third-party readiness metadata, with third-party enablement fixed to `false` for MVP

## Registration Flow

Each widget owns a directory under `src/workbench/widgets/<widget>/` with an `index.ts` that exports its manifest. If the widget renders UI, the manifest owns a single `view` component. `widgetRegistry.ts` imports those manifests into the first-party list, validates them, and returns `RegisteredWidget` records. Invalid manifests follow their failure policy and become disabled or hidden without taking down the workbench.

Widget directories own their renderable surfaces. A widget that can render in a panel, center view, dialog, or popover provides one manifest view and receives the target region as a prop. Multi-surface widgets branch inside that one view instead of exporting one view per surface.

## Regions

Allowed regions are `left-panel`, `right-panel`, `center-view`, `dialog`, `popover`, and `status-bar`.

The widget rails are derived from `getWidgetsForRegion()` instead of hardcoded local lists.

## Graph-Bearing Surfaces

`GraphBearingSurfaceContract` is resolved through `getGraphBearingSurface(widgetId, region)`.

All graph-bearing surfaces expose the same actions through `GraphSurfaceActions`:

- `Set Source`, disabled when already active
- `View Graph`, opening a read-only `GraphPreviewDialog` shell

The current first-party graph-bearing surfaces are Generate, Canvas, and Workflow.

## Failure Isolation

`WidgetFailureBoundary` isolates render failures at widget region boundaries and provides copyable error details. Registration failures are recorded in `WorkbenchState.widgetFailures` and also shown through the shell error surface.
