import type {
  CanvasApplicationOperationStores,
  CanvasApplicationScalarStore,
} from '@workbench/canvas-operations/contracts';

export type CanvasOperationScalarStore<T> = CanvasApplicationScalarStore<T>;
export type CanvasOperationStores = CanvasApplicationOperationStores;

const createScalarStore = <T>(initial: T): CanvasOperationScalarStore<T> => {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set: (next) => {
      if (Object.is(value, next)) {
        return;
      }
      value = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

export const createCanvasOperationStores = (): CanvasOperationStores => ({
  filterSession: createScalarStore(null),
  samSession: createScalarStore(null),
});
