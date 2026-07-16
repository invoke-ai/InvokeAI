# Canvas Context Menu Groups Design

## Goal

Restore the legacy canvas context-menu organization with labeled layer and canvas groups while keeping Delete as the final action.

## Design

- Canvas-surface context menus use Chakra `Menu.ItemGroup` and `Menu.ItemGroupLabel` primitives with the established uppercase, subtle label styling.
- A context menu opened over a layer labels the layer-scoped actions with the singular layer type: Raster Layer, Control Layer, Inpaint Mask, or Regional Guidance.
- Canvas-wide actions are labeled Canvas. Save to Gallery remains in this canvas group and appears before the terminal Delete action.
- Delete remains a separated standalone danger action at the bottom so no non-destructive action appears beneath it.
- A context menu opened over empty canvas space shows only the Canvas group.
- Layer-panel kebab and row context menus remain unchanged because the legacy panel menus were ungrouped.

## Boundaries

- Reuse existing translations for Canvas and singular layer types.
- Preserve all existing handlers, disabled states, submenu behavior, and menu positioning.
- Add no state, effects, dependencies, or action-registry coupling.

## Verification

- Add regression coverage for the surface-menu group model and render order.
- Verify each layer type resolves to the matching legacy label.
- Verify the optional Canvas group precedes the terminal danger section and panel menus remain ungrouped when labels are omitted.
- Run the focused context-menu tests, frontend typecheck, formatting check, and full frontend test suite.
