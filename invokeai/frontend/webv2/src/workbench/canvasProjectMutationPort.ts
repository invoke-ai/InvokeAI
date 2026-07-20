import type { CanvasStateContractV2 } from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import type { CanvasProjectMutation } from './canvasProjectMutations';
import type { WorkbenchCanvasCommands } from './workbenchStore';

export interface CanvasProjectMutationPort {
  getCanvasState(): CanvasStateContractV2 | null;
  subscribe(listener: () => void): () => void;
  dispatch(mutation: CanvasProjectMutation): boolean;
}

export const createCanvasProjectMutationPort = (
  store: {
    commands: { canvas: WorkbenchCanvasCommands };
    getState: () => WorkbenchState;
    subscribe: (listener: () => void) => () => void;
  },
  projectId: string
): CanvasProjectMutationPort => {
  const getCanvasState = (): CanvasStateContractV2 | null =>
    store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;

  return {
    dispatch: (mutation) => store.commands.canvas.apply(projectId, mutation),
    getCanvasState,
    subscribe: store.subscribe,
  };
};
