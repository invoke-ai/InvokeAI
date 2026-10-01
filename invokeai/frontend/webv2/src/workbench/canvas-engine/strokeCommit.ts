import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasMutationContext, EditRefusal } from '@workbench/canvas-engine/controllers/mutationContext';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { HistoryEntry } from '@workbench/canvas-engine/history/history';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { StrokeCommittedEvent, StrokeEdit } from '@workbench/canvas-engine/tools/tool';

import { getDocumentLayer, isNodeAbsent } from '@workbench/canvas-engine/document/documentIndex';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';

export interface CreateStrokeEditsDeps {
  readonly ctx: Pick<CanvasMutationContext, 'applyStep' | 'begin'>;
  readonly layerCache: LayerCacheStore;
  readonly applyImagePatch: ImagePatchApply;
  readonly notifyLayerPainted: (layerId: string) => void;
  readonly markLayerDirty: (layerId: string) => void;
  readonly commitPaintEdit: () => void;
  readonly strokeListeners: ReadonlySet<(event: StrokeCommittedEvent) => void>;
  readonly reportRefusal: (status: EditRefusal) => void;
}

/** The history label for a committed pixel stroke. */
export const strokeCommitLabel = (tool: StrokeCommittedEvent['tool']): string =>
  tool === 'eraser' ? 'Eraser stroke' : tool === 'shape' ? 'Draw shape' : 'Brush stroke';

export interface StrokeEdits {
  /**
   * Admits a live pixel edit before its first pixel changes. Null when it is refused; the refusal is reported once,
   * here or when a later `grow` fails.
   */
  begin(initialBytes?: number): StrokeEdit | null;
}

/**
 * Ordinary strokes paint their preview straight into the layer cache, so publication records the already-applied
 * pixels as one undo step. Auto-created layers compose layer removal/recreation into the same step.
 */
export const createStrokeEdits = (deps: CreateStrokeEditsDeps): StrokeEdits => {
  const { applyImagePatch, ctx, layerCache } = deps;

  const composedEntry = (
    created: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor },
    event: StrokeCommittedEvent
  ): Omit<HistoryEntry, 'label'> => {
    const { afterImageData, dirtyRect, layerId } = event;
    const rect = { ...dirtyRect };
    const removeCreated = (): void =>
      ctx.applyStep({
        accepted: (document) => isNodeAbsent(document, layerId),
        mutation: { ids: [layerId], type: 'removeCanvasLayers' },
      });
    return {
      bytes: event.beforeImageData.data.byteLength + afterImageData.data.byteLength + HISTORY_ENTRY_OVERHEAD_BYTES,
      heldAssetRefs: collectHistoryMediaRefs(created.layer),
      redo: async () => {
        ctx.applyStep({
          accepted: (document) => getDocumentLayer(document, layerId) === created.layer,
          mutation: { anchor: created.anchor, layer: created.layer, type: 'addCanvasLayer' },
          // A fresh empty cache fences async rasterization before the stroke pixels land.
          install: () => {
            layerCache.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 }).stale = false;
          },
        });
        try {
          await applyImagePatch(layerId, rect, afterImageData);
        } catch (error) {
          // A redo that cannot restore the pixels takes the layer back out, so it stays retryable.
          removeCreated();
          throw error;
        }
      },
      undo: () => removeCreated(),
    };
  };

  return {
    begin: (initialBytes = 0) => {
      // The stroke's own gesture starts the edit; it publishes on release, once that gesture has ended.
      const txn = ctx.begin({ gesture: true, historyBytes: initialBytes + HISTORY_ENTRY_OVERHEAD_BYTES });
      if (!('publish' in txn)) {
        deps.reportRefusal(txn.status);
        return null;
      }
      let refusalReported = false;
      return {
        cancel: () => txn.end(),
        commit: (event) => {
          const entry = event.createdLayer
            ? composedEntry(event.createdLayer, event)
            : createImagePatchEntry({
                after: event.afterImageData,
                apply: applyImagePatch,
                before: event.beforeImageData,
                label: strokeCommitLabel(event.tool),
                layerId: event.layerId,
                rect: event.dirtyRect,
              });
          try {
            const result = txn.publish(
              strokeCommitLabel(event.tool),
              {
                notify: () => {
                  deps.notifyLayerPainted(event.layerId);
                  deps.markLayerDirty(event.layerId);
                  deps.commitPaintEdit();
                  for (const listener of deps.strokeListeners) {
                    listener(event);
                  }
                },
              },
              entry,
              { origin: 'system' }
            );
            if (result.status !== 'committed' && !refusalReported) {
              refusalReported = true;
              deps.reportRefusal(result.status === 'over-budget' ? 'over-budget' : 'busy');
            }
            return result.status === 'committed';
          } finally {
            txn.end();
          }
        },
        grow: (bytes) => {
          if (txn.growHistory(bytes)) {
            return true;
          }
          if (!refusalReported) {
            refusalReported = true;
            deps.reportRefusal('over-budget');
          }
          return false;
        },
      };
    },
  };
};
