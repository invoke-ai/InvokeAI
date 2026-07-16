import type { CanvasProjectMutation } from './canvasProjectMutations';
import type { CanvasStateContractV2 } from './types';
import type { WorkbenchStore } from './workbenchStore';

export interface CanvasProjectMutationPort {
  getCanvasState(): CanvasStateContractV2 | null;
  subscribe(listener: () => void): () => void;
  dispatch(mutation: CanvasProjectMutation): boolean;
}

export const createCanvasProjectMutationPort = (
  store: Pick<WorkbenchStore, 'dispatch' | 'getState' | 'subscribe'>,
  projectId: string
): CanvasProjectMutationPort => {
  const getCanvasState = (): CanvasStateContractV2 | null =>
    store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;

  return {
    dispatch: (mutation) => {
      const before = getCanvasState();
      if (!before) {
        return false;
      }
      store.dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' });
      return getCanvasState() !== before;
    },
    getCanvasState,
    subscribe: store.subscribe,
  };
};
