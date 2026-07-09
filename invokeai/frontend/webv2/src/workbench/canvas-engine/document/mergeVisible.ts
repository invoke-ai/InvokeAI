/**
 * Pure planning for "merge visible" (the raster group-header action): folds ALL
 * visible mergeable raster layers together, matching legacy's
 * `compositor.mergeVisibleOfType('raster_layer')`, regardless of how non-raster
 * layers interleave with them in the global z-array.
 *
 * Why interleaving is safe to merge across: since the Task 40 compositor fix the
 * renderer draws layers by GROUP RANK (raster < control < regional guidance <
 * inpaint mask), not by raw global order — a mask or control layer sitting between
 * two rasters in the array never renders between them. Hidden layers of any type
 * do not render at all. So the only layer that can visually separate two raster
 * participants is another raster that RENDERS but cannot participate (a visible
 * locked raster, or a visible parametric shape/text/gradient raster): merging
 * across such a layer would reorder real raster-pass compositing, so it splits
 * the fold into independent runs instead.
 *
 * The engine's `mergeLayerDown` only accepts globally-adjacent pairs, so each
 * planned step may carry a reorder (`orderedIds`) that moves the upper participant
 * to sit directly above its lower partner. That reorder only slides the upper past
 * non-raster layers and/or hidden rasters (render-neutral, see above), and never
 * changes the relative order of visible rasters.
 *
 * Kept pure (no engine, no React) so the planning is unit-testable in node.
 */

import type { CanvasLayerContract } from '@workbench/types';

import { isMergeableRasterLayer } from './sources';

/**
 * True when `layer` renders in the raster compositing pass but cannot join a
 * merge: a visible locked raster, or a visible parametric (shape/text/gradient)
 * raster. Such a layer splits the fold — merging across it would change which
 * pixels composite above/below it.
 */
const isRasterRunBlocker = (layer: CanvasLayerContract): boolean =>
  layer.type === 'raster' && layer.isEnabled && !isMergeableRasterLayer(layer);

/**
 * Partitions the visible mergeable rasters (in global top-to-bottom order) into
 * runs of ids that can fold together. Non-raster layers and hidden rasters never
 * split a run; a rendering non-participant raster does. Only runs with at least
 * two members are returned (a single raster has nothing to merge with).
 */
export const planMergeVisibleRuns = (layers: readonly CanvasLayerContract[]): string[][] => {
  const runs: string[][] = [];
  let current: string[] = [];
  for (const layer of layers) {
    if (isMergeableRasterLayer(layer)) {
      current.push(layer.id);
    } else if (isRasterRunBlocker(layer)) {
      runs.push(current);
      current = [];
    }
  }
  runs.push(current);
  return runs.filter((run) => run.length >= 2);
};

/** Whether the raster group's "merge visible" action has anything to do. */
export const canMergeVisibleRasters = (layers: readonly CanvasLayerContract[]): boolean =>
  planMergeVisibleRuns(layers).length > 0;

/** One step of the merge-visible fold. */
export interface MergeVisibleStep {
  /** The layer merged down (removed by the merge). */
  upperId: string;
  /** The layer it merges into (survives, keeps its transform). */
  lowerId: string;
  /**
   * The full global id order that makes `upperId` directly adjacent above
   * `lowerId`, or null when the pair is already adjacent. Dispatch as a
   * `reorderCanvasLayers` before calling the engine's `mergeLayerDown`.
   */
  orderedIds: string[] | null;
}

/**
 * The next fold step against the CURRENT document: the topmost run's first pair.
 * The caller performs the (optional) reorder + merge, then re-plans against the
 * updated document until this returns null — each merge removes one layer, so the
 * fold strictly terminates.
 */
export const planNextMergeVisibleStep = (layers: readonly CanvasLayerContract[]): MergeVisibleStep | null => {
  const run = planMergeVisibleRuns(layers)[0];
  const upperId = run?.[0];
  const lowerId = run?.[1];
  if (!upperId || !lowerId) {
    return null;
  }
  const ids = layers.map((layer) => layer.id);
  const upperIndex = ids.indexOf(upperId);
  const lowerIndex = ids.indexOf(lowerId);
  if (lowerIndex === upperIndex + 1) {
    return { lowerId, orderedIds: null, upperId };
  }
  const without = ids.filter((id) => id !== upperId);
  without.splice(without.indexOf(lowerId), 0, upperId);
  return { lowerId, orderedIds: without, upperId };
};
