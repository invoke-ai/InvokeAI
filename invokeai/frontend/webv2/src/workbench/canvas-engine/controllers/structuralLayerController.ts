import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { createDocumentPatchEntry } from '@workbench/canvas-engine/history/documentPatch';

export interface StructuralLayerControllerOptions {
  readonly history: History;
  readonly dispatch: (action: CanvasProjectMutation) => void;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly getSelectedLayerIds?: (document: CanvasDocumentContractV2) => readonly string[];
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly now?: () => number;
}

interface NudgeBurst {
  expiresAt: number;
  selectionKey: string;
  origins: readonly { id: string; x: number; y: number }[];
}

const NUDGE_COALESCE_MS = 500;

/** Owns guarded structural document edits and nudge coalescing. */
export class StructuralLayerController {
  private burst: NudgeBurst | null = null;
  private disposed = false;
  private readonly now: () => number;

  constructor(private readonly deps: StructuralLayerControllerOptions) {
    this.now = deps.now ?? Date.now;
  }

  endBurst(): void {
    this.burst = null;
  }

  canCommit(): boolean {
    return !this.disposed && this.deps.canEdit() && !this.deps.isGestureActive();
  }

  commit(label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): boolean {
    if (!this.canCommit()) {
      return false;
    }
    this.endBurst();
    this.deps.dispatch(forward);
    this.deps.history.push(createDocumentPatchEntry({ dispatch: this.deps.dispatch, forward, inverse, label }));
    return true;
  }

  preview(action: CanvasProjectMutation): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    this.deps.dispatch(action);
    return true;
  }

  nudge(dx: number, dy: number): void {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const document = this.deps.getDocument();
    if (!document?.selectedLayerId) {
      return;
    }
    const requested = new Set(this.deps.getSelectedLayerIds?.(document) ?? [document.selectedLayerId]);
    const layers = document.layers.filter((layer) => requested.has(layer.id));
    if (
      layers.length === 0 ||
      layers.length !== requested.size ||
      !requested.has(document.selectedLayerId) ||
      layers.some((layer) => layer.isLocked || !layer.isEnabled)
    ) {
      return;
    }
    const selectionKey = layers.map((layer) => layer.id).join('\0');
    const now = this.now();
    const coalesce = !!this.burst && this.burst.selectionKey === selectionKey && now < this.burst.expiresAt;
    const origins =
      coalesce && this.burst
        ? this.burst.origins
        : layers.map((layer) => ({ id: layer.id, x: layer.transform.x, y: layer.transform.y }));
    const next = layers.map((layer) => ({ id: layer.id, x: layer.transform.x + dx, y: layer.transform.y + dy }));
    const forward: CanvasProjectMutation =
      layers.length === 1
        ? {
            id: layers[0]!.id,
            patch: { transform: { x: next[0]!.x, y: next[0]!.y } },
            type: 'updateCanvasLayer',
          }
        : { type: 'setCanvasLayerPositions', updates: next };
    const inverse: CanvasProjectMutation =
      layers.length === 1
        ? {
            id: layers[0]!.id,
            patch: { transform: { x: origins[0]!.x, y: origins[0]!.y } },
            type: 'updateCanvasLayer',
          }
        : { type: 'setCanvasLayerPositions', updates: origins };
    this.deps.dispatch(forward);
    const entry = createDocumentPatchEntry({ dispatch: this.deps.dispatch, forward, inverse, label: 'Nudge layer' });
    if (coalesce) {
      this.deps.history.amendLast(entry);
    } else {
      this.deps.history.push(entry);
    }
    this.burst = { expiresAt: now + NUDGE_COALESCE_MS, origins, selectionKey };
  }

  dispose(): void {
    this.disposed = true;
    this.endBurst();
  }
}
