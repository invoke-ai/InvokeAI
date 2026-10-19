import { useCallback } from 'react';

import type { CanvasProjectMutation } from './canvasProjectMutations';

import { useActiveProjectId, useWorkbenchCommands } from './WorkbenchContext';

export type CanvasProjectMutationDispatch = (mutation: CanvasProjectMutation) => boolean;

export const useCanvasProjectMutationDispatch = (): CanvasProjectMutationDispatch => {
  const projectId = useActiveProjectId();
  const { canvas } = useWorkbenchCommands();

  return useCallback((mutation: CanvasProjectMutation) => canvas.apply(projectId, mutation), [canvas, projectId]);
};
