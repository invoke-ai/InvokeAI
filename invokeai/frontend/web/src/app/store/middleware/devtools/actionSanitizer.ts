import { AnyAction } from '@reduxjs/toolkit';
import { isAnyGraphBuilt } from 'features/nodes/store/actions';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { Graph } from 'services/api/types';

export const actionSanitizer = <A extends AnyAction>(action: A): A => {
  if (isAnyGraphBuilt(action)) {
    if (action.payload.nodes) {
      const sanitizedNodes: Graph['nodes'] = {};

      return {
        ...action,
        payload: { ...action.payload, nodes: sanitizedNodes },
      };
    }
  }

  if (receivedOpenAPISchema.fulfilled.match(action)) {
    return {
      ...action,
      payload: '<OpenAPI schema omitted>',
    };
  }

  if (nodeTemplatesBuilt.match(action)) {
    return {
      ...action,
      payload: '<Node templates omitted>',
    };
  }

  return action;
};
