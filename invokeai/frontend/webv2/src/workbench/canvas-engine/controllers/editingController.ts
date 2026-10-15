import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasEditGate, CanvasEditGateController } from '@workbench/canvas-engine/editGate';
import type { History } from '@workbench/canvas-engine/history/history';
import type { SelectionState, SelectionStateDeps } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { getSourceBounds, isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import { createCanvasEditGate } from '@workbench/canvas-engine/editGate';
import { roundOut, union } from '@workbench/canvas-engine/math/rect';
import { withSelectionHistory } from '@workbench/canvas-engine/selection/selectionHistory';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';

import { FloatingSelectionController, type FloatingSelectionControllerOptions } from './floatingSelectionController';
import { SelectionImageController, type SelectionImageControllerOptions } from './selectionImageController';
import { SelectionPixelController, type SelectionPixelControllerOptions } from './selectionPixelController';
import { TextEditingController, type TextEditingControllerOptions } from './textEditingController';
import { TransformEditingController, type TransformEditingControllerOptions } from './transformEditingController';

export interface EditingControllerOptions {
  readonly selection: SelectionStateDeps;
  /** Records selection changes; the float folds its own mask move into its entry instead. */
  readonly history: Pick<History, 'admit' | 'isReplaying'>;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly createSelectionState?: (deps: SelectionStateDeps) => SelectionState;
  readonly createEditGate?: () => CanvasEditGateController;
  readonly text: TextEditingControllerOptions;
  readonly transform: TransformEditingControllerOptions;
  readonly selectionPixels: Omit<SelectionPixelControllerOptions, 'selection'>;
  readonly selectionImage: Omit<SelectionImageControllerOptions, 'selection'>;
  readonly floatingSelection: Omit<FloatingSelectionControllerOptions, 'selection'>;
}

/** Owns transient editing state whose lifetime follows one engine instance. */
export class EditingController {
  /** The recording selection: every change through it is an undo step. */
  readonly selection: SelectionState;
  private readonly rawSelection: SelectionState;
  readonly edits: CanvasEditGate;
  readonly text: TextEditingController;
  readonly transform: TransformEditingController;
  readonly selectionPixels: SelectionPixelController;
  readonly selectionImage: SelectionImageController;
  readonly floatingSelection: FloatingSelectionController;
  private readonly editGate: CanvasEditGateController;
  private readonly getDocument: () => CanvasDocumentContractV3 | null;
  private disposed = false;

  constructor(options: EditingControllerOptions) {
    this.rawSelection = (options.createSelectionState ?? createSelectionState)(options.selection);
    this.selection = withSelectionHistory(this.rawSelection, options.history);
    this.getDocument = options.getDocument;
    this.editGate = (options.createEditGate ?? createCanvasEditGate)();
    this.edits = this.editGate;
    this.text = new TextEditingController(options.text);
    this.transform = new TransformEditingController(options.transform);
    this.selectionPixels = new SelectionPixelController({ ...options.selectionPixels, selection: this.selection });
    this.selectionImage = new SelectionImageController({
      ...options.selectionImage,
      selection: this.selection,
    });
    this.floatingSelection = new FloatingSelectionController({
      ...options.floatingSelection,
      selection: this.rawSelection,
    });
  }

  activate(): void {
    if (!this.disposed) {
      this.editGate.activate();
    }
  }

  private selectionDomain(): Rect | null {
    const document = this.getDocument();
    if (!document) {
      return null;
    }
    let bounds: Rect = { ...document.bbox };
    for (const leaf of compileDocumentLeaves(document)) {
      if (leaf.contributionEnabled && isRenderableLayer(leaf.layer)) {
        bounds = union(bounds, getSourceBounds(leaf.layer, document));
      }
    }
    return roundOut(bounds);
  }

  selectAll(): void {
    const domain = this.selectionDomain();
    if (domain) {
      this.selection.selectAll(domain);
    }
  }

  deselect(): void {
    this.selection.clear();
  }

  /** Drops the selection without a history step: the document it belonged to is going away. */
  discardSelection(): void {
    this.rawSelection.clear();
  }

  invertSelection(): void {
    const domain = this.selectionDomain();
    if (domain) {
      this.selection.invert(domain);
    }
  }

  cooldown(): void {
    if (!this.disposed) {
      // Leaving the canvas banks a float like a tool switch, so the flush that follows persists it.
      try {
        this.floatingSelection.commit();
      } catch {
        // A float that cannot land has been put back; cooldown proceeds regardless.
      }
      this.editGate.cooldown();
    }
  }

  invalidateDocument(): void {
    if (!this.disposed) {
      this.editGate.invalidateDocument();
    }
  }

  invalidateProject(): void {
    if (!this.disposed) {
      this.editGate.invalidateProject();
    }
  }

  invalidateLayer(layerId: string): void {
    if (!this.disposed) {
      this.editGate.invalidateLayer(layerId);
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.editGate.dispose();
    this.text.dispose();
    this.transform.dispose();
    this.selectionPixels.dispose();
    this.selectionImage.dispose();
    // Before the selection: disposing a live float puts its pixels back, and
    // that read still needs the selection's mask to be intact.
    this.floatingSelection.dispose();
    this.selection.dispose();
  }
}
