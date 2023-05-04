import { createAction, isAnyOf } from '@reduxjs/toolkit';
import { Graph } from 'services/api';

export const createGraphBuilt = createAction<Graph>('nodes/createGraphBuilt');
export const canvasGraphBuilt = createAction<Graph>('nodes/canvasGraphBuilt');
export const nodesGraphBuilt = createAction<Graph>('nodes/nodesGraphBuilt');

export const isAnyGraphBuilt = isAnyOf(
  createGraphBuilt,
  canvasGraphBuilt,
  nodesGraphBuilt
);
