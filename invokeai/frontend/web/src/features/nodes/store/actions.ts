import { createAction, isAnyOf } from '@reduxjs/toolkit';
import { Graph } from 'services/api';

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
