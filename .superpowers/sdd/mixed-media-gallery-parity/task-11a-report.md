# Task 11A Report: Canvas layer accessibility release blocker

## Status

Complete. The Canvas representative surface no longer reports Axe
`nested-interactive` or `target-size` violations. Layer selection, pointer and
keyboard sorting, visibility controls, thumbnail retry, and locked-state
semantics remain separated and keyboard accessible.

## Root cause

`LayerListItem` spread all dnd-kit attributes and listeners onto the visual
`Row`. dnd-kit therefore gave that container `role="button"` and `tabindex="0"`
while the row still contained visibility, lock, properties, context-menu, and
thumbnail-retry buttons. Axe correctly reported every rendered row as an
interactive control containing focusable descendants.

The visibility control used `ToggleDot`'s compact `h="3" w="3"` button itself
as both the visual dot and hit target. Its measured target was 12×12 pixels,
with insufficient spacing to satisfy Axe's 24-pixel target-size rule.

The repository's working `useWidgetSortable`/`SortableCenterTab` pattern and
local dnd-kit implementation showed the required seam: a dedicated activator
node receives the keyboard listener and sortable semantics, while the sortable
node remains a noninteractive transform/layout container.

## Implementation

- The transformed outer wrapper remains the sortable node.
- The visual row is noninteractive and retains the pointer listener, preserving
  whole-row pointer sorting with the existing six-pixel activation constraint.
- A full-row native `chakra.button` owns mouse, Enter, and Space selection. It is
  a sibling—not an ancestor—of every row control.
- A visible, localized GripVertical button owns dnd-kit attributes, the
  `setActivatorNodeRef`, and keyboard sorting.
- The established modified-Enter and portal-origin keyboard guard remains in
  the keyboard-handle listener.
- Real controls restore pointer events over the otherwise pointer-transparent
  visual layer and keep their existing pointer isolation.
- Thumbnail retry now stops pointerdown before it reaches the row sensor, while
  preserving native focus and click behavior.
- The visibility button has a real 24×24 hit area and a centered 12×12
  pseudo-element dot, preserving the compact visual, semantic tokens, hover
  state, focus ring, tooltip, accessible label, and disabled behavior.
- Added the localized layer-selection accessible label.

No `useEffect`, production module, file rename, `@platform/ui` importer,
migration exception, mock/media behavior, performance baseline, or
architecture-budget change was introduced.

## TDD evidence

The first focused browser RED reproduced both Axe causes:

```text
2 tests failed:
- sortable control contained Toggle visibility
- visibility target measured 12px, expected >=24px
```

The expanded pre-production interaction RED produced seven expected failures
and one pass: independent pointer/Enter/Space selection, row-surface pointer
sorting, and handle-only keyboard sorting did not yet exist; existing visibility
isolation already passed.

After the semantic split, all eight cases passed. The independent review then
identified a real thumbnail-retry gap hidden by the original thumbnail mock.
The test was changed to render the real error-state `LayerThumbnail` with a
faithful preview engine. Its retry-drag case failed by reordering
`first,second` to `second,first`. Adding retry pointerdown isolation made all
nine focused cases pass while also proving focus, click/request, and selection
isolation.

## Verification

All commands ran from `invokeai/frontend/webv2` on the final exact tree.

```text
pnpm lint
  format: passed
  OXC: passed
  tsc: passed
  architecture: 3 files, 34 tests passed

pnpm test
  Test Files 376 passed
  Tests 5005 passed

pnpm test:browser
  Test Files 70 passed
  Tests 358 passed

pnpm test:fixtures
  Tests 10 passed

pnpm test:accessibility
  production build passed
  all 9 representative/keyboard/video reports passed
```

The focused `workbench-canvas-representative` journey passes with every Axe rule
enabled. The `workbench-video-preview-representative` journey also passes with
its sole existing generated-media caption exception.

Invariant scans confirmed `git diff --check` is clean and the diff contains no
new production source file, `useEffect`, gallery/preview/media/mock change,
performance-baseline change, migration exception, or new platform barrel
import.

## Independent review

The first read-only review reported one Important finding and one Minor
finding:

- the real thumbnail retry button did not stop pointerdown from reaching
  row-level DnD;
- the component comment still described the old no-visible-handle design.

Both were corrected. The scoped re-review approved the fixes with no remaining
Critical, Important, or Minor findings.
