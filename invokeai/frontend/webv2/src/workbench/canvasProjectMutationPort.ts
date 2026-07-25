import type { CanvasStateContractV2 } from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import type { CanvasEditIntent, WorkbenchActionOrigin } from './autoRoutePolicy';
import type { CanvasProjectMutation } from './canvasProjectMutations';
import type { WorkbenchCanvasCommands } from './workbenchStore';

export interface CanvasProjectMutationPort {
  getCanvasState(): CanvasStateContractV2 | null;
  subscribe(listener: () => void): () => void;
  dispatch(mutation: CanvasProjectMutation, origin?: WorkbenchActionOrigin): boolean;
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
  const getCanvasState = (): CanvasStateContractV2 | null =>
    store.getState().projects.find((project) => project.id === projectId)?.canvas ?? null;

  return {
    commitEdit: (intent) => store.commands.canvas.commitEdit(projectId, intent),
    dispatch: (mutation, origin) => store.commands.canvas.apply(projectId, mutation, origin),
    getCanvasState,
    subscribe: store.subscribe,
  };
};
