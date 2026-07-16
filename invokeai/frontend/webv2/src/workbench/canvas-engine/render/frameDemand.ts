import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2 } from '@workbench/types';

import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, transformBounds, union } from '@workbench/canvas-engine/math/rect';

export interface FrameDemandInput {
  readonly document: CanvasDocumentContractV2;
  readonly isolationLayerIds?: ReadonlySet<string>;
  readonly liveCacheRects?: ReadonlyMap<string, Rect>;
  readonly transformOverrides?: ReadonlyMap<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >;
  readonly viewport: Rect;
}

/** Calculates the enabled raster caches whose transformed pixels intersect the next frame. */
export const calculateActiveFrameLayerIds = ({
  document,
  isolationLayerIds,
  liveCacheRects,
  transformOverrides,
  viewport,
}: FrameDemandInput): Set<string> => {
  const active = new Set<string>();
  for (const layer of document.layers) {
    if (!layer.isEnabled || !renderableSourceOf(layer) || (isolationLayerIds && !isolationLayerIds.has(layer.id))) {
      continue;
    }
    const sourceRect = getSourceContentRect(layer, document);
    const liveRect = liveCacheRects?.get(layer.id);
    const localRect =
      liveRect && !isEmpty(liveRect) ? (isEmpty(sourceRect) ? liveRect : union(sourceRect, liveRect)) : sourceRect;
    const override = transformOverrides?.get(layer.id);
    const transform = override
      ? {
          rotation: override.rotation ?? layer.transform.rotation,
          scaleX: override.scaleX ?? layer.transform.scaleX,
          scaleY: override.scaleY ?? layer.transform.scaleY,
          x: override.x,
          y: override.y,
        }
      : layer.transform;
    const matrix = fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);
    if (intersect(transformBounds(matrix, localRect), viewport)) {
      active.add(layer.id);
    }
  }
  return active;
};
