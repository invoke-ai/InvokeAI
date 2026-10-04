import type { CanvasOperationCapability, CanvasOperationImplementation } from './contracts';

const operationsByEngine = new WeakMap<object, CanvasOperationImplementation>();

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
