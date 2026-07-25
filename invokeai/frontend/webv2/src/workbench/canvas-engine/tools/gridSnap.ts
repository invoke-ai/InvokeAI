/**
 * The grid-snapping gate for the tools that position LAYERS (move, transform).
 *
 * The canvas has exactly one grid: `stores.bboxGrid`, the model grid that the bbox
 * tool snaps to and that `overlayRenderer` actually paints. Layer positions snap to
 * that same grid, so a layer lands on the lines the user can see — there is no
 * second, invisible positional grid (legacy's 64/8 pair does not apply here; its
 * drawn grid was decoupled from the bbox grid).
 *
 * The math itself lives in `math/snapping` (`snapToGrid`, `snapMovedPoint`), which
 * stays pure and store-free; this module only answers "which grid, right now".
 *
 * Zero React, zero import-time side effects.
 */

import type { ToolContext } from './tool';

/**
 * The grid layer positions should snap to right now, or `0` for no snap —
 * `snapToGrid` treats `grid <= 0` as "leave the value alone", so callers never
 * need a second boolean.
 *
 * `bypass` is the per-gesture escape hatch (alt, matching the bbox tool). Alt
 * reaches the tool rather than switching to the color picker because
 * `pointerPipeline.beginTempTool` refuses to start a temp-tool hold while a
 * gesture is active — the same mechanism the bbox tool's alt bypass relies on.
 */
export const positionGrid = (ctx: ToolContext, bypass: boolean): number =>
  !bypass && ctx.stores.snapToGrid.get() ? ctx.stores.bboxGrid.get() : 0;
