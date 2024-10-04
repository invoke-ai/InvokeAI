import { createAction, isAnyOf } from '@reduxjs/toolkit';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import type { Graph, GraphAndWorkflowResponse } from 'services/api/types';

const textToImageGraphBuilt = createAction<Graph>('nodes/textToImageGraphBuilt');
const imageToImageGraphBuilt = createAction<Graph>('nodes/imageToImageGraphBuilt');
const canvasGraphBuilt = createAction<Graph>('nodes/canvasGraphBuilt');
const nodesGraphBuilt = createAction<Graph>('nodes/nodesGraphBuilt');

export const isAnyGraphBuilt = isAnyOf(
  textToImageGraphBuilt,
  imageToImageGraphBuilt,
  canvasGraphBuilt,
  nodesGraphBuilt
);

export const workflowLoadRequested = createAction<{
  data: GraphAndWorkflowResponse;
  asCopy: boolean;
}>('nodes/workflowLoadRequested');

export const updateAllNodesRequested = createAction('nodes/updateAllNodesRequested');

export const workflowLoaded = createAction<WorkflowV3>('workflow/workflowLoaded');
