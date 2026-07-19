import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasEditGate, CanvasEditGateController } from '@workbench/canvas-engine/editGate';
import type { SelectionState, SelectionStateDeps } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

import { getSourceBounds, isRenderableLayer } from '@workbench/canvas-engine/document/sources';
import { createCanvasEditGate } from '@workbench/canvas-engine/editGate';
import { roundOut, union } from '@workbench/canvas-engine/math/rect';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';

import { SelectionImageController, type SelectionImageControllerOptions } from './selectionImageController';
import { SelectionPixelController, type SelectionPixelControllerOptions } from './selectionPixelController';
import { TextEditingController, type TextEditingControllerOptions } from './textEditingController';
import { TransformEditingController, type TransformEditingControllerOptions } from './transformEditingController';

export interface EditingControllerOptions<Permit = unknown, Owner = symbol> {
  readonly selection: SelectionStateDeps;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly createSelectionState?: (deps: SelectionStateDeps) => SelectionState;
  readonly createEditGate?: () => CanvasEditGateController;
  readonly text: TextEditingControllerOptions;
  readonly transform: TransformEditingControllerOptions;
  readonly selectionPixels: Omit<SelectionPixelControllerOptions, 'selection'>;
  readonly selectionImage: Omit<SelectionImageControllerOptions<Permit, Owner>, 'selection'>;
}

/** Owns transient editing state whose lifetime follows one engine instance. */
export class EditingController<Permit = unknown, Owner = symbol> {
  readonly selection: SelectionState;
  readonly edits: CanvasEditGate;
  readonly text: TextEditingController;
  readonly transform: TransformEditingController;
  readonly selectionPixels: SelectionPixelController;
  readonly selectionImage: SelectionImageController<Permit, Owner>;
  private readonly editGate: CanvasEditGateController;
  private readonly getDocument: () => CanvasDocumentContractV2 | null;
  private disposed = false;

  constructor(options: EditingControllerOptions<Permit, Owner>) {
    this.selection = (options.createSelectionState ?? createSelectionState)(options.selection);
    this.getDocument = options.getDocument;
    this.editGate = (options.createEditGate ?? createCanvasEditGate)();
    this.edits = this.editGate;
    this.text = new TextEditingController(options.text);
    this.transform = new TransformEditingController(options.transform);
    this.selectionPixels = new SelectionPixelController({ ...options.selectionPixels, selection: this.selection });
    this.selectionImage = new SelectionImageController<Permit, Owner>({
      ...options.selectionImage,
      selection: this.selection,
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
    for (const layer of document.layers) {
      if (isRenderableLayer(layer)) {
        bounds = union(bounds, getSourceBounds(layer, document));
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

  invertSelection(): void {
    const domain = this.selectionDomain();
    if (domain) {
      this.selection.invert(domain);
    }
  }

  cooldown(): void {
    if (!this.disposed) {
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
    this.selection.dispose();
  }
}
