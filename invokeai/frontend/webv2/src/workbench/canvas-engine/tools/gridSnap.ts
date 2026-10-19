/**
 * Layer positioning uses the same model grid the bbox tool and overlay show. Pure snapping math remains in
 * math/snapping; this module resolves current settings.
 */

import type { ToolContext } from './tool';

/**
 * Returns active grid or zero to disable snapping. Alt bypass reaches active gestures because temporary picker
 * switching is suppressed mid-gesture.
 */
export const positionGrid = (ctx: ToolContext, bypass: boolean): number =>
  !bypass && ctx.stores.snapToGrid.get() ? ctx.stores.bboxGrid.get() : 0;
