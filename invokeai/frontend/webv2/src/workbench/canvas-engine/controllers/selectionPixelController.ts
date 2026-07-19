import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { ControlPixelEditTransaction } from '@workbench/canvas-engine/tools/tool';
import type { Rect } from '@workbench/canvas-engine/types';

import { getSourceBounds } from '@workbench/canvas-engine/document/sources';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { intersect, isEmpty, roundOut } from '@workbench/canvas-engine/math/rect';
import { eraseMaskedRegion, fillMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';

type PixelTarget =
  | { kind: 'raster'; layerId: string; transparencyLocked: boolean }
  | { kind: 'control'; transaction: ControlPixelEditTransaction; transparencyLocked: false };

export interface SelectionPixelControllerOptions {
  readonly selection: SelectionState;
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly history: History;
  readonly applyImagePatch: ImagePatchApply;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly beginControlEdit: (layerId: string) => ControlPixelEditTransaction | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly getFillColor: () => string;
  readonly endBurst: () => void;
  readonly deleteDerived: (layerId: string) => void;
  readonly invalidateLayer: (layerId: string) => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
}

const imageDataEqual = (left: ImageData, right: ImageData): boolean => {
  if (left.width !== right.width || left.height !== right.height || left.data.length !== right.data.length) {
    return false;
  }
  for (let index = 0; index < left.data.length; index += 1) {
    if (left.data[index] !== right.data[index]) {
      return false;
    }
  }
  return true;
};

/** Owns selection-driven fill/erase pixel transactions. */
export class SelectionPixelController {
  private disposed = false;

  constructor(private readonly deps: SelectionPixelControllerOptions) {}

  private target(): PixelTarget | null {
    const document = this.deps.getDocument();
    if (!document?.selectedLayerId) {
      return null;
    }
    const layer = document.layers.find((candidate) => candidate.id === document.selectedLayerId);
    if (layer?.type === 'raster' && layer.source.type === 'paint' && !layer.isLocked && layer.isEnabled) {
      return { kind: 'raster', layerId: layer.id, transparencyLocked: layer.isTransparencyLocked === true };
    }
    if (layer?.type === 'control') {
      const transaction = this.deps.beginControlEdit(layer.id);
      return transaction ? { kind: 'control', transaction, transparencyLocked: false } : null;
    }
    return null;
  }

  run(kind: 'fill' | 'erase'): void {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const document = this.deps.getDocument();
    const placedMask = this.deps.selection.mask();
    const bounds = this.deps.selection.bounds();
    if (!document || !placedMask || !bounds) {
      return;
    }
    const selectionRect = roundOut(bounds);
    const selectedLayer = document.layers.find((candidate) => candidate.id === document.selectedLayerId);
    if (
      kind === 'erase' &&
      selectedLayer?.type === 'control' &&
      !intersect(selectionRect, roundOut(getSourceBounds(selectedLayer, document)))
    ) {
      return;
    }
    const target = this.target();
    if (!target) {
      return;
    }
    const cancelControl = (): void => {
      if (target.kind === 'control') {
        target.transaction.cancel();
      }
    };
    if (kind === 'erase' && target.transparencyLocked) {
      cancelControl();
      return;
    }
    const layerId = target.kind === 'control' ? target.transaction.layerId : target.layerId;
    let before: ImageData | null = null;
    let editOrigin: { x: number; y: number } | null = null;
    let editRect: Rect | null = null;
    let editSurface: RasterSurface | null = null;
    let commitStarted = false;
    let rollbackStarted = false;
    let growthSnapshot:
      | {
          hasPublishedPixels: boolean;
          lastUsed: number;
          pixels: ImageData | null;
          rect: Rect;
          stale: boolean;
          surface: RasterSurface;
          version: number;
        }
      | null
      | undefined;
    const rollback = (): void => {
      if (target.kind !== 'control' || rollbackStarted) {
        return;
      }
      rollbackStarted = true;
      try {
        if (growthSnapshot !== undefined) {
          if (growthSnapshot === null) {
            this.deps.layers.delete(layerId);
          } else {
            const current = this.deps.layers.get(layerId);
            if (current) {
              growthSnapshot.surface.resize(growthSnapshot.rect.width, growthSnapshot.rect.height);
              if (growthSnapshot.pixels) {
                growthSnapshot.surface.ctx.putImageData(growthSnapshot.pixels, 0, 0);
              }
              current.hasPublishedPixels = growthSnapshot.hasPublishedPixels;
              current.lastUsed = growthSnapshot.lastUsed;
              current.rect = { ...growthSnapshot.rect };
              current.stale = growthSnapshot.stale;
              current.surface = growthSnapshot.surface;
              current.version = growthSnapshot.version;
            }
          }
          this.deps.deleteDerived(layerId);
          this.deps.invalidateLayer(layerId);
        } else if (before && editOrigin && editRect && editSurface) {
          editSurface.ctx.putImageData(before, editRect.x - editOrigin.x, editRect.y - editOrigin.y);
          this.deps.deleteDerived(layerId);
          this.deps.invalidateLayer(layerId);
        }
      } finally {
        target.transaction.cancel();
      }
    };
    try {
      if (target.kind === 'control' && kind === 'fill') {
        const existing = this.deps.layers.get(layerId);
        const needsGrowth =
          !existing ||
          isEmpty(existing.rect) ||
          selectionRect.x < existing.rect.x ||
          selectionRect.y < existing.rect.y ||
          selectionRect.x + selectionRect.width > existing.rect.x + existing.rect.width ||
          selectionRect.y + selectionRect.height > existing.rect.y + existing.rect.height;
        if (needsGrowth) {
          growthSnapshot = existing
            ? {
                hasPublishedPixels: existing.hasPublishedPixels,
                lastUsed: existing.lastUsed,
                pixels: isEmpty(existing.rect)
                  ? null
                  : existing.surface.ctx.getImageData(0, 0, existing.rect.width, existing.rect.height),
                rect: { ...existing.rect },
                stale: existing.stale,
                surface: existing.surface,
                version: existing.version,
              }
            : null;
        }
      }
      let rect: Rect | null;
      let entry;
      if (kind === 'fill' && !target.transparencyLocked) {
        rect = selectionRect;
        entry = this.deps.layers.growToRect(layerId, selectionRect);
      } else {
        const existing = this.deps.layers.get(layerId);
        if (!existing || isEmpty(existing.rect)) {
          cancelControl();
          return;
        }
        rect = intersect(selectionRect, existing.rect);
        entry = existing;
      }
      if (!rect || isEmpty(rect)) {
        cancelControl();
        return;
      }
      this.deps.endBurst();
      const surface = entry.surface;
      const origin = { x: entry.rect.x, y: entry.rect.y };
      before = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
      editOrigin = origin;
      editRect = rect;
      editSurface = surface;
      if (kind === 'fill') {
        fillMaskedRegion({
          backend: this.deps.backend,
          color: this.deps.getFillColor(),
          composite: target.transparencyLocked ? 'source-atop' : 'source-over',
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          rect,
          target: surface,
          targetOrigin: origin,
        });
      } else {
        eraseMaskedRegion({
          backend: this.deps.backend,
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          rect,
          target: surface,
          targetOrigin: origin,
        });
      }
      const after = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
      const label = kind === 'fill' ? 'Fill selection' : 'Erase selection';
      if (target.kind === 'control') {
        if (imageDataEqual(before, after)) {
          rollback();
          return;
        }
        commitStarted = true;
        target.transaction.commitPatch(label, { after, before, rect });
      } else {
        this.deps.notifyPainted(target.layerId);
        this.deps.markDirty(target.layerId);
        if (!this.deps.history.isApplying()) {
          this.deps.history.push(
            createImagePatchEntry({
              after,
              apply: this.deps.applyImagePatch,
              before,
              label,
              layerId: target.layerId,
              rect,
            })
          );
        }
      }
    } catch (error) {
      if (target.kind !== 'control' || commitStarted) {
        throw error;
      }
      rollback();
      throw error;
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
