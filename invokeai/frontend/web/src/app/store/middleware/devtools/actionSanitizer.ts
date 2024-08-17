import type { UnknownAction } from '@reduxjs/toolkit';
import { isAnyGraphBuilt } from 'features/nodes/store/actions';
import { appInfoApi } from 'services/api/endpoints/appInfo';
import type { Graph } from 'services/api/types';

export const actionSanitizer = <A extends UnknownAction>(action: A): A => {
  if (isAnyGraphBuilt(action)) {
    if (action.payload.nodes) {
      const sanitizedNodes: Graph['nodes'] = {};

      return {
        ...action,
        payload: { ...action.payload, nodes: sanitizedNodes },
      };
    }
  }

  if (appInfoApi.endpoints.getOpenAPISchema.matchFulfilled(action)) {
    return {
      ...action,
      payload: '<OpenAPI schema omitted>',
    };
  }

  return action;
};
