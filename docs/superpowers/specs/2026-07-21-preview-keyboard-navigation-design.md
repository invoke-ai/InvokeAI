# Preview Keyboard Navigation Design

## Goal

Make left/right keyboard navigation in the preview widget deterministic: one keypress moves exactly one position, and the current generating placeholder is reachable from and can return to completed images.

## Behavior contract

- The preview navigation sequence contains the current board's completed images plus only the active generating placeholder.
- The sequence follows the gallery's configured image order. In descending order, the placeholder occupies the newest position; in ascending order, it occupies the latest chronological position.
- Starred-first ordering remains consistent with the gallery: the placeholder follows the leading starred block.
- Left and right clamp at sequence boundaries and never wrap or skip.
- Moving to an image selects it and pauses live-follow through the existing gallery command.
- Moving to the placeholder enables live-follow through the existing account command.
- Comparison mode retains its current behavior and does not expose the placeholder.

## Architecture

Introduce a small pure navigation model near the preview widget. It derives a discriminated sequence of image and placeholder items, resolves the current cursor from selected-image/live-follow state, and returns the adjacent target for an offset of `-1` or `1`.

The preview view owns one navigation action that applies the returned target. Both saved-image and live-preview branches use the same focusable keyboard boundary and handler. The handler prevents the browser default and stops propagation so the widget-scoped hotkey runtime cannot execute the same arrow event a second time.

No additional persisted cursor is introduced. Gallery selection and the existing live-follow preference remain the sources of truth.

## Failure handling

If the selected image is absent from the loaded board, the active placeholder is unavailable, or the cursor is already at a boundary, navigation is a no-op. A queue update that removes the placeholder naturally derives a sequence without it; no synchronization effect is needed.

## Verification

- Pure unit tests cover one-step movement, clamping, sort direction, starred-first placement, and transitions in both directions between an image and the active placeholder.
- A focused browser/component test dispatches one arrow event through the preview boundary and proves only one selection command occurs, guarding against double handling by the hotkey runtime.
- Existing preview, gallery-state, typecheck, and formatting checks run after implementation.
