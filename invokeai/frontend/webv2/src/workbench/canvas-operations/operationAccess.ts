import type { CanvasDocumentContractV3, CanvasEngine } from '@workbench/canvas-engine/api';

import { getDocumentLeaves } from '@workbench/canvas-engine/api';

import type { CanvasOperationCapability, CanvasOperationImplementation } from './contracts';

const operationsByEngine = new WeakMap<object, CanvasOperationImplementation>();
let lookupCanvasEngine: ((projectId: string) => CanvasEngine | undefined) | undefined;

/** The default registry binds this when Canvas composition loads; a non-owning lookup never loads or creates it. */
export const bindCanvasEngineLookup = (lookup: (projectId: string) => CanvasEngine | undefined): void => {
  lookupCanvasEngine = lookup;
};

/** Queries the default registry without constructing an engine or acquiring a lease. */
export const getCanvasEngine = (projectId: string): CanvasEngine | undefined => lookupCanvasEngine?.(projectId);

/** Whether the canvas holds nothing yet: no raster, control or regional layer, and no mask with coverage. */
export const isBlankCanvasDocument = (document: CanvasDocumentContractV3 | null): boolean =>
  getDocumentLeaves(document).every((layer) => layer.type === 'inpaint_mask' && layer.mask.bitmap === null);

export const getCanvasOperationImplementation = (engine: object): CanvasOperationImplementation => {
  const operations = operationsByEngine.get(engine);
  if (!operations) {
    throw new Error('Canvas application operations are unavailable for this engine.');
  }
  return operations;
};

export const getCanvasOperations = (engine: object): CanvasOperationCapability =>
  getCanvasOperationImplementation(engine);

/** Canvas composition hook; callers consume only `getCanvasOperations`. */
export const attachCanvasOperations = (engine: object, operations: CanvasOperationImplementation): void => {
  operationsByEngine.set(engine, operations);
};
