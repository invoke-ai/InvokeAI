import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasControlLayerContract } from '@workbench/types';

import type { HistoryEntry } from './history';

export interface LayerPixelSnapshot {
  layer: CanvasControlLayerContract;
  rect: Rect;
  /** Null represents the exact pixel state of a zero-width or zero-height cache. */
  pixels: ImageData | null;
}

export type LayerPixelSnapshotApply = (snapshot: LayerPixelSnapshot) => void;

export interface CreateLayerSnapshotEntryOptions {
  before: LayerPixelSnapshot;
  after: LayerPixelSnapshot;
  label: string;
  apply: LayerPixelSnapshotApply;
}

const assertSnapshot = (snapshot: LayerPixelSnapshot, which: 'before' | 'after'): void => {
  if (snapshot.rect.width === 0 || snapshot.rect.height === 0) {
    if (snapshot.pixels !== null) {
      throw new Error(`layerSnapshot: ${which} empty rect requires null pixels`);
    }
    return;
  }
  if (snapshot.pixels === null) {
    throw new Error(`layerSnapshot: ${which} non-empty rect requires pixels`);
  }
  if (snapshot.pixels.width !== snapshot.rect.width || snapshot.pixels.height !== snapshot.rect.height) {
    throw new Error(`layerSnapshot: ${which} pixels do not match rect`);
  }
};

export const createLayerSnapshotEntry = ({
  after,
  apply,
  before,
  label,
}: CreateLayerSnapshotEntryOptions): HistoryEntry => {
  if (before.layer.id !== after.layer.id) {
    throw new Error('layerSnapshot: before/after layer ids differ');
  }
  assertSnapshot(before, 'before');
  assertSnapshot(after, 'after');
  const beforeSnapshot: LayerPixelSnapshot = {
    layer: structuredClone(before.layer),
    pixels: before.pixels,
    rect: { ...before.rect },
  };
  const afterSnapshot: LayerPixelSnapshot = {
    layer: structuredClone(after.layer),
    pixels: after.pixels,
    rect: { ...after.rect },
  };
  return {
    bytes: (before.pixels?.data.byteLength ?? 0) + (after.pixels?.data.byteLength ?? 0) + 256,
    label,
    redo: () => apply(afterSnapshot),
    replayFailureAtomic: true,
    undo: () => apply(beforeSnapshot),
  };
};
