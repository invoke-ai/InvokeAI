import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createDocumentPatchEntry } from '@workbench/canvas-engine/history/documentPatch';

export interface StructuralLayerControllerOptions {
  readonly history: History;
  readonly dispatch: (action: CanvasProjectMutation) => void;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly now?: () => number;
}

interface NudgeBurst {
  expiresAt: number;
  layerId: string;
  origin: { x: number; y: number };
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
    const layer = document?.layers.find((candidate) => candidate.id === document.selectedLayerId);
    if (!document?.selectedLayerId || !layer || layer.isLocked || !layer.isEnabled) {
      return;
    }
    const next = { x: layer.transform.x + dx, y: layer.transform.y + dy };
    const now = this.now();
    const coalesce = !!this.burst && this.burst.layerId === layer.id && now < this.burst.expiresAt;
    const origin = coalesce && this.burst ? this.burst.origin : { x: layer.transform.x, y: layer.transform.y };
    const forward: CanvasProjectMutation = {
      id: layer.id,
      patch: { transform: next },
      type: 'updateCanvasLayer',
    };
    const inverse: CanvasProjectMutation = {
      id: layer.id,
      patch: { transform: origin },
      type: 'updateCanvasLayer',
    };
    this.deps.dispatch(forward);
    const entry = createDocumentPatchEntry({ dispatch: this.deps.dispatch, forward, inverse, label: 'Nudge layer' });
    if (coalesce) {
      this.deps.history.amendLast(entry);
    } else {
      this.deps.history.push(entry);
    }
    this.burst = { expiresAt: now + NUDGE_COALESCE_MS, layerId: layer.id, origin };
  }

  dispose(): void {
    this.disposed = true;
    this.endBurst();
  }
}
