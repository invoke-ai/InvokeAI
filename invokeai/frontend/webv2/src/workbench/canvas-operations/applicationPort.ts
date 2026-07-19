import type { CanvasApplicationPort } from '@workbench/canvas-operations/contracts';

import { runUtilityGraph } from '@features/queue/utility';
import { socketHub } from '@platform/transport/socketHub';
import { uploadCanvasImage } from '@workbench/canvas-operations/backend/canvasImages';
import {
  createFilterOperationCoordinator,
  type FilterOperationCoordinatorDeps,
} from '@workbench/canvas-operations/FilterOperationCoordinator';
import { createCanvasOperationController } from '@workbench/canvas-operations/operationController';
import { createCanvasOperationStores } from '@workbench/canvas-operations/operationStores';
import { createSelectObjectOperationCoordinator } from '@workbench/canvas-operations/SelectObjectCoordinator';
import { getSelectedModelBase } from '@workbench/widgets/layers/selectedModel';

export const canvasApplicationPort: CanvasApplicationPort = {
  createFilterCoordinator: (deps) => createFilterOperationCoordinator(deps as FilterOperationCoordinatorDeps),
  createOperationController: (deps) => createCanvasOperationController(deps),
  createOperationStores: createCanvasOperationStores,
  createSelectObjectCoordinator: createSelectObjectOperationCoordinator,
  getSelectedModelBase: (state, projectId) => {
    const project = state.projects.find((candidate) => candidate.id === projectId);
    return project ? getSelectedModelBase(project) : null;
  },
  runGraph: (options) => runUtilityGraph({ ...options, hub: socketHub }),
  uploadImage: (blob, options) => uploadCanvasImage(blob, options),
};
