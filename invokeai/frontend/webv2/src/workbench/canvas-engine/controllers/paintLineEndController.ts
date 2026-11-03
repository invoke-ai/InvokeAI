import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { PaintLineEnds } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { applyToPoint, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { layerTransformMatrix } from '@workbench/canvas-engine/transform/transformMath';

export interface PaintLineEndControllerOptions {
  /** The layer as the document holds it now, or null when it is gone. */
  readonly getLayer: (layerId: string) => CanvasLayerContract | null;
}

interface LineEnd {
  /** Layer-local, in the space of `layer.transform`. */
  readonly point: Vec2;
  /** The contract last seen for the layer, which `point` is relative to. */
  readonly layer: CanvasLayerContract;
}

const sameTransform = (a: LayerTransform, b: LayerTransform): boolean =>
  a.x === b.x && a.y === b.y && a.rotation === b.rotation && a.scaleX === b.scaleX && a.scaleY === b.scaleY;

/**
 * Where the last committed brush or eraser stroke ended on each layer, for Shift-click straight lines. Points are kept
 * layer-local, so a change of transform alone (a move, nudge, Fit to frame or transform parameter) carries them with
 * the layer. A change that rewrites the layer's content and transform together (crop, conversion, a baked float or
 * result) keeps the point where the content stays in the document, mapping it from the previous transform into the
 * new one. An applied Transform on a paint layer rewrites only its pixels, so it says where they went with
 * {@link rebase}. A point the new transform cannot hold is forgotten.
 */
export class PaintLineEndController implements PaintLineEnds {
  private readonly ends = new Map<string, LineEnd>();

  constructor(private readonly deps: PaintLineEndControllerOptions) {}

  get(layerId: string): Vec2 | undefined {
    // Catches up with a change no layer event reported (a layer moved between stacks reports a reorder).
    this.onLayerChanged(layerId);
    return this.ends.get(layerId)?.point;
  }

  set(layerId: string, point: Vec2): void {
    const layer = this.deps.getLayer(layerId);
    if (layer) {
      this.ends.set(layerId, { layer, point: { ...point } });
    }
  }

  /** Follows one document change of `layerId`. */
  onLayerChanged(layerId: string): void {
    const end = this.ends.get(layerId);
    if (!end) {
      return;
    }
    const layer = this.deps.getLayer(layerId);
    if (!layer) {
      this.ends.delete(layerId);
      return;
    }
    if (layer === end.layer) {
      return;
    }
    const rebaked =
      !sameTransform(layer.transform, end.layer.transform) &&
      (layer.type !== end.layer.type ||
        !areJsonValuesStructurallyEqual(renderableSourceOf(layer), renderableSourceOf(end.layer)));
    if (!rebaked) {
      this.ends.set(layerId, { layer, point: end.point });
      return;
    }
    this.map(layerId, end.point, end.layer.transform, layer);
  }

  /**
   * The layer's pixels were redrawn through `from`, the transform they were shown at, into its current local space
   * (an applied Transform on a paint layer); the point follows them.
   */
  rebase(layerId: string, from: LayerTransform): void {
    this.onLayerChanged(layerId);
    const end = this.ends.get(layerId);
    if (end) {
      this.map(layerId, end.point, from, end.layer);
    }
  }

  /** Stores `point`, local to `from`, as the same document point local to `layer`'s transform. */
  private map(layerId: string, point: Vec2, from: LayerTransform, layer: CanvasLayerContract): void {
    const toLocal = invert(layerTransformMatrix(layer.transform));
    if (!toLocal) {
      this.ends.delete(layerId);
      return;
    }
    this.ends.set(layerId, { layer, point: applyToPoint(multiply(toLocal, layerTransformMatrix(from)), point) });
  }

  delete(layerId: string): void {
    this.ends.delete(layerId);
  }

  clear(): void {
    this.ends.clear();
  }

  dispose(): void {
    this.ends.clear();
  }
}
