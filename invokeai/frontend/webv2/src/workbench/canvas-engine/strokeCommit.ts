import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History, HistoryEntry } from '@workbench/canvas-engine/history/history';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';
import type { Rect } from '@workbench/canvas-engine/types';

import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';

export interface CreateStrokeCommitDeps {
  readonly history: History;
  readonly layerCache: LayerCacheStore;
  readonly applyImagePatch: ImagePatchApply;
  readonly dispatchCanvasMutation: (mutation: CanvasProjectMutation) => boolean;
  readonly endNudgeBurst: () => void;
  readonly notifyLayerPainted: (layerId: string) => void;
  readonly markLayerDirty: (layerId: string) => void;
  readonly commitPaintEdit: () => void;
  readonly strokeListeners: ReadonlySet<(event: StrokeCommittedEvent) => void>;
}

export interface StrokeCommit {
  /** Records one committed tool stroke: persistence, history, then subscriber fanout. */
  commitOrdinaryStroke(event: StrokeCommittedEvent): void;
}

/**
 * Turns a committed stroke into its durable consequences.
 *
 * A stroke that auto-created its own layer cannot be undone as a plain pixel
 * patch — undo has to remove the layer too — so those get a composed history
 * entry that pairs the layer mutation with the pixel write. Ordinary strokes
 * take the pixel-patch path.
 */
export const createStrokeCommit = (deps: CreateStrokeCommitDeps): StrokeCommit => {
  const { applyImagePatch, dispatchCanvasMutation, history, layerCache } = deps;

  const createComposedPaintEntry = (
    created: { layer: CanvasLayerContract; index: number },
    event: StrokeCommittedEvent,
    label: string
  ): HistoryEntry => {
    const { afterImageData, dirtyRect, layerId } = event;
    const rect: Rect = { height: dirtyRect.height, width: dirtyRect.width, x: dirtyRect.x, y: dirtyRect.y };
    const bytes = event.beforeImageData.data.byteLength + afterImageData.data.byteLength + 256;
    return {
      bytes,
      label,
      redo: () => {
        dispatchCanvasMutation({ index: created.index, layer: created.layer, type: 'addCanvasLayer' });
        // Re-create an EMPTY cache marked fresh so the async rasterize pass can't
        // clobber the restored stroke; `applyImagePatch` grows it to the stroke's
        // content bounds and writes the `after` pixels.
        const entry = layerCache.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
        entry.stale = false;
        applyImagePatch(layerId, rect, afterImageData);
      },
      undo: () => {
        dispatchCanvasMutation({ ids: [layerId], type: 'removeCanvasLayers' });
      },
    };
  };

  return {
    commitOrdinaryStroke: (event) => {
      // A fresh stroke ends any open nudge-coalescing burst.
      deps.endNudgeBurst();
      deps.notifyLayerPainted(event.layerId);
      // Persistence first: mark the layer dirty so a debounced upload fires even
      // when no external subscriber is attached.
      deps.markLayerDirty(event.layerId);
      // Record the edit on the engine-owned history. Guarded against re-entrancy:
      // an undo/redo replay routes pixels through `applyImagePatch`, not a fresh
      // stroke, so this never fires during apply — the guard is belt-and-braces.
      if (!history.isApplying()) {
        deps.commitPaintEdit();
        const label = event.tool === 'eraser' ? 'Eraser stroke' : 'Brush stroke';
        history.push(
          event.createdLayer
            ? createComposedPaintEntry(event.createdLayer, event, label)
            : createImagePatchEntry({
                after: event.afterImageData,
                apply: applyImagePatch,
                before: event.beforeImageData,
                label,
                layerId: event.layerId,
                rect: event.dirtyRect,
              })
        );
      }
      for (const listener of deps.strokeListeners) {
        listener(event);
      }
    },
  };
};
