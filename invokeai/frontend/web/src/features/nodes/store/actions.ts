import { createAction, isAnyOf } from '@reduxjs/toolkit';
import { Graph } from 'services/api/types';
import { Workflow } from '../types/types';

export const textToImageGraphBuilt = createAction<Graph>(
  'nodes/textToImageGraphBuilt'
);
export const imageToImageGraphBuilt = createAction<Graph>(
  'nodes/imageToImageGraphBuilt'
);
export const canvasGraphBuilt = createAction<Graph>('nodes/canvasGraphBuilt');
export const nodesGraphBuilt = createAction<Graph>('nodes/nodesGraphBuilt');

export const isAnyGraphBuilt = isAnyOf(
  textToImageGraphBuilt,
  imageToImageGraphBuilt,
  canvasGraphBuilt,
  nodesGraphBuilt
);

export const workflowLoadRequested = createAction<Workflow>(
  'nodes/workflowLoadRequested'
);
