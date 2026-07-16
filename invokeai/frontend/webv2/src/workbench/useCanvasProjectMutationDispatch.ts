import { useCallback } from 'react';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { useActiveProjectId, useWorkbenchStore } from './WorkbenchContext';

export type CanvasProjectMutationDispatch = (mutation: CanvasProjectMutation) => boolean;

export const useCanvasProjectMutationDispatch = (): CanvasProjectMutationDispatch => {
  const projectId = useActiveProjectId();
  const store = useWorkbenchStore();

  return useCallback(
    (mutation: CanvasProjectMutation) => {
      const before = store.getState().projects.find((project) => project.id === projectId)?.canvas;
      if (!before) {
        return false;
      }
      store.dispatch({ mutation, projectId, type: 'applyCanvasProjectMutation' });
      return store.getState().projects.find((project) => project.id === projectId)?.canvas !== before;
    },
    [projectId, store]
  );
};
