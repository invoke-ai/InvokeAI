import type {
  CanvasEditIntent,
  CanvasMutationOrigin,
  CanvasProjectMutation,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import type { WorkbenchCanvasCommands } from './workbenchStore';

export interface CanvasProjectMutationPort {
  getCanvasState(): CanvasStateContractV3 | null;
  subscribe(listener: () => void): () => void;
  dispatch(mutation: CanvasProjectMutation, origin?: CanvasMutationOrigin): boolean;
  commitEdit(intent: CanvasEditIntent): void;
}

export const createCanvasProjectMutationPort = (
  store: {
    commands: { canvas: WorkbenchCanvasCommands };
    getState: () => WorkbenchState;
    subscribe: (listener: () => void) => () => void;
  },
  projectId: string
): CanvasProjectMutationPort => {
  const getCanvasState = (): CanvasStateContractV3 | null =>
    store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;

  return {
    commitEdit: (intent) => store.commands.canvas.commitEdit(projectId, intent),
    dispatch: (mutation, origin) => store.commands.canvas.apply(projectId, mutation, origin),
    getCanvasState,
    subscribe: store.subscribe,
  };
};
